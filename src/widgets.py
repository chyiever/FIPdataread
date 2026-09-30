"""Background workers and the interactive time-domain plot widget.

This module holds the three Qt classes that the main window needs but that are
not themselves part of the main-window behaviour mixins:

* ``LoadWaveformWorker``   - reads and concatenates waveform files off the GUI thread.
* ``SVMPredictionWorker``  - runs the sliding-window SVM predictor off the GUI thread.
* ``TimePlotWidget``       - the drag/zoom/select plot used for Plot 1, Plot 2 and
                             the compact t-f time plot.

They are separated from ``main_window`` so that the behaviour mixins can reference
them without importing the composed ``MainWindow`` class (which would be a
circular import).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore

from data_access import load_waveforms_concatenated
from models import InteractionMode
from plotting import make_pen
from processing import (
    compute_short_time_svm_predictions,
    load_sliding_window_svm_predictor,
)

class LoadWaveformWorker(QtCore.QObject):
    """QThread worker that loads waveform data in the background.
    """

    finished = QtCore.pyqtSignal(int, object)
    failed = QtCore.pyqtSignal(int, str)

    def __init__(self, task_id: int, paths: list[Path]) -> None:
        """Store the task id and the ordered list of paths to load.
        """

        super().__init__()
        self._task_id = task_id
        self._paths = paths

    @QtCore.pyqtSlot()
    def run(self) -> None:
        """Load the concatenated waveform and emit the result signal.
        """

        try:
            waveform = load_waveforms_concatenated(self._paths)
        except Exception as exc:
            self.failed.emit(self._task_id, str(exc))
            return
        self.finished.emit(self._task_id, waveform)


class SVMPredictionWorker(QtCore.QObject):
    """QThread worker that computes sliding-window SVM predictions in the background.
    """

    finished = QtCore.pyqtSignal(int, object, object)
    failed = QtCore.pyqtSignal(int, str)

    def __init__(self, task_id: int, model_directory: Path, values: np.ndarray, sample_rate: float) -> None:
        """Copy the input array and store predictor parameters.

        The values are copied so later mutation of the caller's array cannot change the
        data mid-prediction.
        """

        super().__init__()
        self._task_id = task_id
        self._model_directory = model_directory
        self._values = np.asarray(values, dtype=np.float64).copy()
        self._sample_rate = float(sample_rate)

    @QtCore.pyqtSlot()
    def run(self) -> None:
        """Load the SVM predictor, run the sliding-window prediction, emit the result.

        Uses a 40 ms window with 50% hop, matching the parameters documented in the
        README and the development log.
        """

        try:
            predictor = load_sliding_window_svm_predictor(str(self._model_directory))
            centers, predictions = compute_short_time_svm_predictions(
                self._values,
                self._sample_rate,
                predictor=predictor,
                window_seconds=0.04,
                hop_ratio=0.5,
            )
        except Exception as exc:
            self.failed.emit(self._task_id, str(exc))
            return
        self.finished.emit(self._task_id, centers, predictions)


class TimePlotWidget(pg.PlotWidget):
    """Custom ``PlotWidget`` that uses the absolute-time axis.
    """

    windowSelected = QtCore.pyqtSignal(int, int)
    clearSelectionRequested = QtCore.pyqtSignal()
    arrivalMarkRequested = QtCore.pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        """Initialise interaction state and default to Zoom mode.
        """

        super().__init__(*args, **kwargs)
        self._interaction_mode = InteractionMode.ZOOM
        self._selection_active = False
        self._selection_start_x = 0.0
        self._selection_region: Optional[pg.LinearRegionItem] = None
        self._data_length = 0
        self._sample_rate = 1.0
        self._min_window_samples = 1
        self._pan_active = False
        self._pan_last_scene_pos: Optional[QtCore.QPointF] = None

    def set_interaction_mode(self, mode: InteractionMode) -> None:
        """Switch between RectangleMode (Zoom) and PanMode (Window PSD).
        """

        self._interaction_mode = mode
        self.getViewBox().setMouseMode(
            pg.ViewBox.RectMode if mode == InteractionMode.ZOOM else pg.ViewBox.PanMode
        )

    def set_data_context(self, data_length: int, sample_rate: float, min_window_seconds: float) -> None:
        """Tell the plot the data length, sample rate and minimum window.

        The minimum window (in samples) is enforced on selection so a PSD can never be
        computed from fewer samples than the user configured.
        """

        self._data_length = max(0, int(data_length))
        self._sample_rate = max(float(sample_rate), 1.0)
        self._min_window_samples = max(1, int(round(min_window_seconds * self._sample_rate)))

    def set_selection_region(self, start_index: int, end_index: int) -> None:
        """Show (or move) the yellow selection region over [lo, hi].
        """

        lo, hi = sorted((start_index, end_index))
        if self._selection_region is None:
            self._selection_region = pg.LinearRegionItem(
                values=(lo, hi),
                brush=(255, 215, 0, 60),
                pen=make_pen("#CC9900", 1),
                movable=False,
            )
            self.addItem(self._selection_region)
        else:
            self._selection_region.setRegion((lo, hi))

    def clear_selection_region(self) -> None:
        """Remove the selection region item if one is present.
        """

        if self._selection_region is not None:
            self.removeItem(self._selection_region)
            self._selection_region = None

    def mousePressEvent(self, event):
        """Handle press: arrival marking, panning, selection start or passthrough.
        """

        if event.button() == QtCore.Qt.RightButton and bool(event.modifiers() & QtCore.Qt.ControlModifier):
            point = self.plotItem.vb.mapSceneToView(self.mapToScene(event.pos()))
            self.arrivalMarkRequested.emit(self._clamp_x(point.x()))
            event.accept()
            return

        if event.button() == QtCore.Qt.MiddleButton or (
            event.button() == QtCore.Qt.LeftButton
            and bool(event.modifiers() & QtCore.Qt.ShiftModifier)
        ):
            self._pan_active = True
            self._pan_last_scene_pos = self.mapToScene(event.pos())
            event.accept()
            return

        if self._interaction_mode == InteractionMode.WINDOW_PSD:
            if event.button() == QtCore.Qt.RightButton:
                self.clearSelectionRequested.emit()
                event.accept()
                return
            if event.button() == QtCore.Qt.LeftButton:
                point = self.plotItem.vb.mapSceneToView(self.mapToScene(event.pos()))
                self._selection_start_x = point.x()
                self._selection_active = True
                bounded = self._clamp_x(self._selection_start_x)
                self.set_selection_region(int(round(bounded)), int(round(bounded)))
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle drag: pan the view or grow the active selection.
        """

        if self._pan_active and self._pan_last_scene_pos is not None:
            current_scene_pos = self.mapToScene(event.pos())
            previous_view = self.plotItem.vb.mapSceneToView(self._pan_last_scene_pos)
            current_view = self.plotItem.vb.mapSceneToView(current_scene_pos)
            delta_x = previous_view.x() - current_view.x()
            if delta_x != 0.0:
                self.plotItem.vb.translateBy(x=delta_x, y=0.0)
            self._pan_last_scene_pos = current_scene_pos
            event.accept()
            return

        if self._interaction_mode == InteractionMode.WINDOW_PSD and self._selection_active:
            point = self.plotItem.vb.mapSceneToView(self.mapToScene(event.pos()))
            current_x = self._clamp_x(point.x())
            start_x = self._clamp_x(self._selection_start_x)
            self.set_selection_region(int(round(start_x)), int(round(current_x)))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle release: stop panning or commit the selection window.
        """

        if self._pan_active and (
            event.button() == QtCore.Qt.MiddleButton or event.button() == QtCore.Qt.LeftButton
        ):
            self._pan_active = False
            self._pan_last_scene_pos = None
            event.accept()
            return

        if (
            self._interaction_mode == InteractionMode.WINDOW_PSD
            and self._selection_active
            and event.button() == QtCore.Qt.LeftButton
        ):
            self._selection_active = False
            point = self.plotItem.vb.mapSceneToView(self.mapToScene(event.pos()))
            start_x = int(round(self._clamp_x(self._selection_start_x)))
            end_x = int(round(self._clamp_x(point.x())))
            lo, hi = sorted((start_x, end_x))
            if hi - lo < self._min_window_samples:
                hi = min(self._data_length - 1, lo + self._min_window_samples)
            if hi > lo:
                self.set_selection_region(lo, hi)
                self.windowSelected.emit(lo, hi)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _clamp_x(self, value: float) -> float:
        """Clamp a view x-coordinate into the valid sample-index range.
        """

        if self._data_length <= 1:
            return 0.0
        return min(max(float(value), 0.0), float(self._data_length - 1))

