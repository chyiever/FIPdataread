from __future__ import annotations

import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtMultimedia, QtWidgets

from config import UI_DEFAULTS
from data_access import (
    build_export_tdms_name,
    list_data_files,
    load_waveform,
    paginate_files,
    save_tdms_waveform,
    save_wav_waveform,
)
from models import (
    PAGE_SIZE,
    FileRecord,
    FilterMode,
    InteractionMode,
    LoadedWaveform,
    SortField,
)
from plotting import AbsoluteTimeAxis, LogFrequencyAxis, configure_plot_widget, create_colormap, make_pen
from processing import (
    apply_display_filter,
    compute_time_frequency_map,
    compute_short_time_energy_ratio,
    compute_short_time_svm_predictions,
    compute_window_psd,
    load_sliding_window_svm_predictor,
    prepare_audio_waveform,
    validate_filter,
)

FEATURE_MODE_SVM = "svm_prediction"
FEATURE_MODE_ENERGY = "short_time_energy"
TF_MODE_PSD = "psd"
TF_MODE_AMPLITUDE = "amplitude"
TF_SCALE_LOG = "log"
TF_SCALE_LINEAR = "linear"


class LoadWaveformWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(int, object)
    failed = QtCore.pyqtSignal(int, str)

    def __init__(self, task_id: int, path: Path) -> None:
        super().__init__()
        self._task_id = task_id
        self._path = path

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            waveform = load_waveform(self._path)
        except Exception as exc:
            self.failed.emit(self._task_id, str(exc))
            return
        self.finished.emit(self._task_id, waveform)


class SVMPredictionWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(int, object, object)
    failed = QtCore.pyqtSignal(int, str)

    def __init__(self, task_id: int, model_directory: Path, values: np.ndarray, sample_rate: float) -> None:
        super().__init__()
        self._task_id = task_id
        self._model_directory = model_directory
        self._values = np.asarray(values, dtype=np.float64).copy()
        self._sample_rate = float(sample_rate)

    @QtCore.pyqtSlot()
    def run(self) -> None:
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
    windowSelected = QtCore.pyqtSignal(int, int)
    clearSelectionRequested = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
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
        self._interaction_mode = mode
        self.getViewBox().setMouseMode(
            pg.ViewBox.RectMode if mode == InteractionMode.ZOOM else pg.ViewBox.PanMode
        )

    def set_data_context(self, data_length: int, sample_rate: float, min_window_seconds: float) -> None:
        self._data_length = max(0, int(data_length))
        self._sample_rate = max(float(sample_rate), 1.0)
        self._min_window_samples = max(1, int(round(min_window_seconds * self._sample_rate)))

    def set_selection_region(self, start_index: int, end_index: int) -> None:
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
        if self._selection_region is not None:
            self.removeItem(self._selection_region)
            self._selection_region = None

    def mousePressEvent(self, event):
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
        if self._data_length <= 1:
            return 0.0
        return min(max(float(value), 0.0), float(self._data_length - 1))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FIPread")
        self.resize(1500, 920)

        self._source_files: list[FileRecord] = []
        self._all_files: list[FileRecord] = []
        self._page_index = 0
        self._current_waveform: Optional[LoadedWaveform] = None
        self._current_display_values = np.array([], dtype=np.float64)
        self._view_history: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self._last_view_state: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
        self._suspend_history = False
        self._syncing_scrollbar = False
        self._scroll_shortcuts: list[QtWidgets.QShortcut] = []
        self._fixed_psd_enabled = False
        self._fixed_psd_anchor_ratio = 0.5
        self._fixed_psd_window_samples = 0
        self._svm_model_directory = Path(__file__).resolve().parent.parent / "models" / "saved_models"
        self._load_thread: Optional[QtCore.QThread] = None
        self._load_worker: Optional[LoadWaveformWorker] = None
        self._load_task_id = 0
        self._prediction_thread: Optional[QtCore.QThread] = None
        self._prediction_worker: Optional[SVMPredictionWorker] = None
        self._prediction_task_id = 0
        self._audio_temp_path: Optional[Path] = None
        self._audio_player = QtMultimedia.QMediaPlayer(self)
        self._audio_player.setVolume(100)
        self._audio_path_auto_managed = True
        self._syncing_time_tf_x = False
        self._tf_freq_hz = np.array([], dtype=np.float64)
        self._tf_time_centers = np.array([], dtype=np.float64)
        self._tf_base_values = np.array([], dtype=np.float64)
        self._tf_base_mode = TF_MODE_PSD
        self._tf_log_freq_bounds: Optional[tuple[float, float]] = None
        self._tf_default_color_min = UI_DEFAULTS.display.tf_color_min
        self._tf_color_min_user_override = False
        self._updating_tf_color_spins = False
        self._clamping_time_x_range = False
        self._tf_side_panel_width = 126

        self._build_ui()
        self._bind_events()
        self._refresh_file_list()

    def _refresh_default_audio_path(self) -> None:
        if self._audio_path_auto_managed:
            self.audio_path_edit.setText(str(self._default_audio_path()))

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)
        self.logo_label = QtWidgets.QLabel()
        self.logo_label.setFixedSize(56, 56)
        self.logo_label.setAlignment(QtCore.Qt.AlignCenter)
        logo_path = Path(__file__).resolve().parent.parent / "logo.png"
        if logo_path.exists():
            pixmap = QtGui.QPixmap(str(logo_path))
            if not pixmap.isNull():
                self.logo_label.setPixmap(
                    pixmap.scaled(
                        self.logo_label.size(),
                        QtCore.Qt.KeepAspectRatio,
                        QtCore.Qt.SmoothTransformation,
                    )
                )
        self.app_title_label = QtWidgets.QLabel("FIP数据处理软件")
        self.app_title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_font = QtGui.QFont("SimSun", 24, QtGui.QFont.Bold)
        self.app_title_label.setFont(title_font)
        header_layout.addStretch(1)
        header_layout.addWidget(self.logo_label)
        header_layout.addWidget(self.app_title_label)
        header_layout.addStretch(1)
        main_layout.addLayout(header_layout)

        control_scroll = QtWidgets.QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        control_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        control_panel = QtWidgets.QFrame()
        control_panel.setMinimumWidth(360)
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(30)
        control_scroll.setWidget(control_panel)

        directory_group = QtWidgets.QGroupBox("Files Management")
        directory_layout = QtWidgets.QGridLayout(directory_group)
        self.directory_edit = QtWidgets.QLineEdit(str(Path.cwd() / "data"))
        self.browse_button = QtWidgets.QPushButton("Browse")
        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.export_directory_edit = QtWidgets.QLineEdit(str(Path.cwd() / "exports"))
        self.export_browse_button = QtWidgets.QPushButton("Export Dir")
        self.export_visible_button = QtWidgets.QPushButton("Export Visible Raw Data")
        directory_layout.addWidget(self.directory_edit, 0, 0, 1, 2)
        directory_layout.addWidget(self.browse_button, 0, 2)
        directory_layout.addWidget(QtWidgets.QLabel("Export Path"), 1, 0)
        directory_layout.addWidget(self.export_directory_edit, 1, 1)
        directory_layout.addWidget(self.export_browse_button, 1, 2)
        directory_layout.addWidget(self.refresh_button, 2, 0)
        directory_layout.addWidget(self.export_visible_button, 2, 1, 1, 2)

        sort_group = QtWidgets.QGroupBox("File List")
        sort_layout = QtWidgets.QGridLayout(sort_group)
        self.sort_field_combo = QtWidgets.QComboBox()
        self.sort_field_combo.addItem("Name", SortField.NAME)
        self.sort_field_combo.addItem("Modified Time", SortField.MTIME)
        self.sort_order_combo = QtWidgets.QComboBox()
        self.sort_order_combo.addItem("Ascending", True)
        self.sort_order_combo.addItem("Descending", False)
        self.file_list = QtWidgets.QListWidget()
        self.file_list.setMinimumHeight(240)
        self.file_list.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.page_info_label = QtWidgets.QLabel("Page 0 / 0 | Total 0")
        self.home_button = QtWidgets.QPushButton("Home")
        self.prev_button = QtWidgets.QPushButton(f"Previous {PAGE_SIZE}")
        self.next_button = QtWidgets.QPushButton(f"Next {PAGE_SIZE}")
        self.end_button = QtWidgets.QPushButton("End")
        self.page_jump_spin = QtWidgets.QSpinBox()
        self.page_jump_spin.setMinimum(1)
        self.page_jump_spin.setMaximum(1)
        self.jump_button = QtWidgets.QPushButton("Go To Page")
        self.amplitude_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.amplitude_threshold_spin.setDecimals(6)
        self.amplitude_threshold_spin.setRange(0.0, 1e12)
        self.amplitude_threshold_spin.setValue(UI_DEFAULTS.file.amplitude_threshold)
        self.threshold_filter_button = QtWidgets.QPushButton("Threshold Filter")
        sort_layout.addWidget(QtWidgets.QLabel("Sort By"), 0, 0)
        sort_layout.addWidget(self.sort_field_combo, 0, 1)
        sort_layout.addWidget(QtWidgets.QLabel("Order"), 0, 2)
        sort_layout.addWidget(self.sort_order_combo, 0, 3)
        sort_layout.addWidget(self.file_list, 1, 0, 1, 4)
        sort_layout.addWidget(self.page_info_label, 2, 0, 1, 4)
        sort_layout.addWidget(self.home_button, 3, 0)
        sort_layout.addWidget(self.prev_button, 3, 1)
        sort_layout.addWidget(self.next_button, 3, 2)
        sort_layout.addWidget(self.end_button, 3, 3)
        sort_layout.addWidget(QtWidgets.QLabel("Page"), 4, 0)
        sort_layout.addWidget(self.page_jump_spin, 4, 1)
        sort_layout.addWidget(self.jump_button, 4, 2)
        sort_layout.addWidget(QtWidgets.QLabel("Amplitude Threshold"), 5, 0, 1, 2)
        sort_layout.addWidget(self.amplitude_threshold_spin, 5, 2)
        sort_layout.addWidget(self.threshold_filter_button, 5, 3)

        filter_group = QtWidgets.QWidget()
        filter_layout = QtWidgets.QGridLayout(filter_group)
        self.filter_enabled_checkbox = QtWidgets.QCheckBox("Enable Filter")
        self.filter_enabled_checkbox.setChecked(UI_DEFAULTS.display.filter_enabled)
        self.filter_mode_combo = QtWidgets.QComboBox()
        self.filter_mode_combo.addItem(FilterMode.BANDPASS.value, FilterMode.BANDPASS)
        self.filter_mode_combo.addItem(FilterMode.HIGHPASS.value, FilterMode.HIGHPASS)
        self.filter_mode_combo.addItem(FilterMode.LOWPASS.value, FilterMode.LOWPASS)
        self.low_cut_spin = QtWidgets.QDoubleSpinBox()
        self.low_cut_spin.setDecimals(1)
        self.low_cut_spin.setRange(0.0, 10_000_000.0)
        self.low_cut_spin.setValue(UI_DEFAULTS.display.low_cut_hz)
        self.high_cut_spin = QtWidgets.QDoubleSpinBox()
        self.high_cut_spin.setDecimals(1)
        self.high_cut_spin.setRange(0.0, 10_000_000.0)
        self.high_cut_spin.setValue(UI_DEFAULTS.display.high_cut_hz)
        self.filter_mode_combo.setCurrentIndex(UI_DEFAULTS.display.filter_mode_index)
        self.apply_filter_button = QtWidgets.QPushButton("Apply Display Filter")
        self.y_min_spin = QtWidgets.QDoubleSpinBox()
        self.y_min_spin.setDecimals(3)
        self.y_min_spin.setRange(-1e12, 1e12)
        self.y_min_spin.setValue(UI_DEFAULTS.display.phase_y_min)
        self.y_max_spin = QtWidgets.QDoubleSpinBox()
        self.y_max_spin.setDecimals(3)
        self.y_max_spin.setRange(-1e12, 1e12)
        self.y_max_spin.setValue(UI_DEFAULTS.display.phase_y_max)
        self.psd_y_min_spin = QtWidgets.QSpinBox()
        self.psd_y_min_spin.setRange(-200, 100)
        self.psd_y_min_spin.setValue(UI_DEFAULTS.display.psd_y_min)
        self.psd_y_max_spin = QtWidgets.QSpinBox()
        self.psd_y_max_spin.setRange(-200, 100)
        self.psd_y_max_spin.setValue(UI_DEFAULTS.display.psd_y_max)
        self.tf_mode_combo = QtWidgets.QComboBox()
        self.tf_mode_combo.addItem("PSD", TF_MODE_PSD)
        self.tf_mode_combo.addItem("Amplitude", TF_MODE_AMPLITUDE)
        self.tf_mode_combo.setCurrentIndex(UI_DEFAULTS.display.tf_mode_index)
        self.tf_value_scale_combo = QtWidgets.QComboBox()
        self.tf_value_scale_combo.addItem("Log", TF_SCALE_LOG)
        self.tf_value_scale_combo.addItem("Linear", TF_SCALE_LINEAR)
        self.tf_value_scale_combo.setCurrentIndex(UI_DEFAULTS.display.tf_value_scale_index)
        self.tf_window_spin = QtWidgets.QDoubleSpinBox()
        self.tf_window_spin.setDecimals(4)
        self.tf_window_spin.setRange(0.0001, 10.0)
        self.tf_window_spin.setValue(UI_DEFAULTS.display.tf_window_seconds)
        self.tf_overlap_spin = QtWidgets.QDoubleSpinBox()
        self.tf_overlap_spin.setDecimals(1)
        self.tf_overlap_spin.setRange(0.0, 95.0)
        self.tf_overlap_spin.setValue(UI_DEFAULTS.display.tf_overlap_percent)
        self.tf_y_min_spin = QtWidgets.QDoubleSpinBox()
        self.tf_y_min_spin.setDecimals(1)
        self.tf_y_min_spin.setRange(0.0, 10_000_000.0)
        self.tf_y_min_spin.setValue(UI_DEFAULTS.display.tf_y_min_hz)
        self.tf_y_max_spin = QtWidgets.QDoubleSpinBox()
        self.tf_y_max_spin.setDecimals(1)
        self.tf_y_max_spin.setRange(0.0, 10_000_000.0)
        self.tf_y_max_spin.setValue(UI_DEFAULTS.display.tf_y_max_hz)
        self.tf_colormap_combo = QtWidgets.QComboBox()
        self.tf_colormap_combo.setEditable(True)
        self.tf_colormap_combo.addItems(["jet", "hsv", "seismic", "viridis", "plasma", "magma", "inferno", "turbo"])
        self.tf_colormap_combo.setCurrentText(UI_DEFAULTS.display.tf_colormap)
        self.tf_color_auto_checkbox = QtWidgets.QCheckBox("t-f Color Auto")
        self.tf_color_auto_checkbox.setChecked(UI_DEFAULTS.display.tf_color_auto)
        self.tf_color_min_spin = QtWidgets.QDoubleSpinBox()
        self.tf_color_min_spin.setDecimals(3)
        self.tf_color_min_spin.setRange(-1e12, 1e12)
        self.tf_color_min_spin.setValue(UI_DEFAULTS.display.tf_color_min)
        self.tf_color_max_spin = QtWidgets.QDoubleSpinBox()
        self.tf_color_max_spin.setDecimals(3)
        self.tf_color_max_spin.setRange(-1e12, 1e12)
        self.tf_color_max_spin.setValue(UI_DEFAULTS.display.tf_color_max)
        self.apply_y_range_button = QtWidgets.QPushButton("Apply Y Range")
        self.apply_tf_button = QtWidgets.QPushButton("Apply t-f Params")
        filter_layout.addWidget(QtWidgets.QLabel("Filter Mode"), 0, 0)
        filter_layout.addWidget(self.filter_mode_combo, 0, 1)
        filter_layout.addWidget(self.filter_enabled_checkbox, 0, 2, 1, 2)
        filter_layout.addWidget(QtWidgets.QLabel("Low Cut (Hz)"), 1, 0)
        filter_layout.addWidget(self.low_cut_spin, 1, 1)
        filter_layout.addWidget(QtWidgets.QLabel("High Cut (Hz)"), 1, 2)
        filter_layout.addWidget(self.high_cut_spin, 1, 3)
        filter_layout.addWidget(QtWidgets.QLabel("Phase Y Min"), 2, 0)
        filter_layout.addWidget(self.y_min_spin, 2, 1)
        filter_layout.addWidget(QtWidgets.QLabel("Phase Y Max"), 2, 2)
        filter_layout.addWidget(self.y_max_spin, 2, 3)
        filter_layout.addWidget(QtWidgets.QLabel("PSD Y Min"), 3, 0)
        filter_layout.addWidget(self.psd_y_min_spin, 3, 1)
        filter_layout.addWidget(QtWidgets.QLabel("PSD Y Max"), 3, 2)
        filter_layout.addWidget(self.psd_y_max_spin, 3, 3)
        filter_layout.addWidget(QtWidgets.QLabel("t-f Mode"), 4, 0)
        filter_layout.addWidget(self.tf_mode_combo, 4, 1)
        filter_layout.addWidget(QtWidgets.QLabel("t-f Value"), 4, 2)
        filter_layout.addWidget(self.tf_value_scale_combo, 4, 3)
        filter_layout.addWidget(QtWidgets.QLabel("t-f Window (s)"), 5, 0)
        filter_layout.addWidget(self.tf_window_spin, 5, 1)
        filter_layout.addWidget(QtWidgets.QLabel("t-f Overlap (%)"), 5, 2)
        filter_layout.addWidget(self.tf_overlap_spin, 5, 3)
        filter_layout.addWidget(QtWidgets.QLabel("t-f Y Min (Hz)"), 6, 0)
        filter_layout.addWidget(self.tf_y_min_spin, 6, 1)
        filter_layout.addWidget(QtWidgets.QLabel("t-f Y Max (Hz)"), 6, 2)
        filter_layout.addWidget(self.tf_y_max_spin, 6, 3)
        filter_layout.addWidget(QtWidgets.QLabel("t-f Colormap"), 7, 0)
        filter_layout.addWidget(self.tf_colormap_combo, 7, 1)
        filter_layout.addWidget(self.tf_color_auto_checkbox, 7, 2, 1, 2)
        filter_layout.addWidget(QtWidgets.QLabel("t-f Color Min"), 8, 0)
        filter_layout.addWidget(self.tf_color_min_spin, 8, 1)
        filter_layout.addWidget(QtWidgets.QLabel("t-f Color Max"), 8, 2)
        filter_layout.addWidget(self.tf_color_max_spin, 8, 3)
        filter_layout.addWidget(self.apply_filter_button, 9, 0, 1, 4)
        filter_layout.addWidget(self.apply_y_range_button, 10, 0, 1, 4)
        filter_layout.addWidget(self.apply_tf_button, 11, 0, 1, 4)

        feature_group = QtWidgets.QWidget()
        feature_layout = QtWidgets.QGridLayout(feature_group)
        self.feature_num_low_spin = QtWidgets.QDoubleSpinBox()
        self.feature_num_low_spin.setDecimals(1)
        self.feature_num_low_spin.setRange(0.0, 1e9)
        self.feature_num_low_spin.setValue(UI_DEFAULTS.feature.band1_low_hz)
        self.feature_num_high_spin = QtWidgets.QDoubleSpinBox()
        self.feature_num_high_spin.setDecimals(1)
        self.feature_num_high_spin.setRange(0.0, 1e9)
        self.feature_num_high_spin.setValue(UI_DEFAULTS.feature.band1_high_hz)
        self.feature_den_low_spin = QtWidgets.QDoubleSpinBox()
        self.feature_den_low_spin.setDecimals(1)
        self.feature_den_low_spin.setRange(0.0, 1e9)
        self.feature_den_low_spin.setValue(UI_DEFAULTS.feature.band2_low_hz)
        self.feature_den_high_spin = QtWidgets.QDoubleSpinBox()
        self.feature_den_high_spin.setDecimals(1)
        self.feature_den_high_spin.setRange(0.0, 1e9)
        self.feature_den_high_spin.setValue(UI_DEFAULTS.feature.band2_high_hz)
        self.feature_window_spin = QtWidgets.QDoubleSpinBox()
        self.feature_window_spin.setDecimals(4)
        self.feature_window_spin.setRange(0.0001, 10.0)
        self.feature_window_spin.setValue(UI_DEFAULTS.feature.window_seconds)
        self.feature_step_spin = QtWidgets.QDoubleSpinBox()
        self.feature_step_spin.setDecimals(1)
        self.feature_step_spin.setRange(1.0, 100.0)
        self.feature_step_spin.setValue(UI_DEFAULTS.feature.step_percent)
        self.feature_amp_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.feature_amp_threshold_spin.setDecimals(6)
        self.feature_amp_threshold_spin.setRange(0.0, 1e12)
        self.feature_amp_threshold_spin.setValue(UI_DEFAULTS.feature.amplitude_gate)
        self.feature_y_min_spin = QtWidgets.QDoubleSpinBox()
        self.feature_y_min_spin.setDecimals(6)
        self.feature_y_min_spin.setRange(-1e12, 1e12)
        self.feature_y_min_spin.setValue(UI_DEFAULTS.feature.y_min)
        self.feature_y_max_spin = QtWidgets.QDoubleSpinBox()
        self.feature_y_max_spin.setDecimals(6)
        self.feature_y_max_spin.setRange(-1e12, 1e12)
        self.feature_y_max_spin.setValue(UI_DEFAULTS.feature.y_max)
        self.feature_apply_y_range_button = QtWidgets.QPushButton("Apply Feature Y Range")
        self.feature_apply_button = QtWidgets.QPushButton("Apply Feature Params")
        feature_layout.addWidget(QtWidgets.QLabel("Band 1 Low (Hz)"), 0, 0)
        feature_layout.addWidget(self.feature_num_low_spin, 0, 1)
        feature_layout.addWidget(QtWidgets.QLabel("Band 1 High (Hz)"), 1, 0)
        feature_layout.addWidget(self.feature_num_high_spin, 1, 1)
        feature_layout.addWidget(QtWidgets.QLabel("Band 2 Low (Hz)"), 2, 0)
        feature_layout.addWidget(self.feature_den_low_spin, 2, 1)
        feature_layout.addWidget(QtWidgets.QLabel("Band 2 High (Hz)"), 3, 0)
        feature_layout.addWidget(self.feature_den_high_spin, 3, 1)
        feature_layout.addWidget(QtWidgets.QLabel("Window (s)"), 4, 0)
        feature_layout.addWidget(self.feature_window_spin, 4, 1)
        feature_layout.addWidget(QtWidgets.QLabel("Step (% of window)"), 5, 0)
        feature_layout.addWidget(self.feature_step_spin, 5, 1)
        feature_layout.addWidget(QtWidgets.QLabel("Amplitude Gate"), 6, 0)
        feature_layout.addWidget(self.feature_amp_threshold_spin, 6, 1)
        feature_layout.addWidget(QtWidgets.QLabel("Feature Y Min"), 7, 0)
        feature_layout.addWidget(self.feature_y_min_spin, 7, 1)
        feature_layout.addWidget(QtWidgets.QLabel("Feature Y Max"), 8, 0)
        feature_layout.addWidget(self.feature_y_max_spin, 8, 1)
        feature_layout.addWidget(self.feature_apply_y_range_button, 9, 0, 1, 2)
        feature_layout.addWidget(self.feature_apply_button, 10, 0, 1, 2)

        audio_group = QtWidgets.QWidget()
        audio_layout = QtWidgets.QGridLayout(audio_group)
        self.audio_path_edit = QtWidgets.QLineEdit(str(self._default_audio_path()))
        self.audio_path_browse_button = QtWidgets.QPushButton("Audio File")
        self.audio_downsample_spin = QtWidgets.QSpinBox()
        self.audio_downsample_spin.setRange(1, 1000000)
        self.audio_downsample_spin.setValue(UI_DEFAULTS.audio.downsample_factor)
        self.play_audio_button = QtWidgets.QPushButton("Play")
        self.stop_audio_button = QtWidgets.QPushButton("Stop")
        self.replay_audio_button = QtWidgets.QPushButton("Replay")
        self.export_audio_button = QtWidgets.QPushButton("Export Visible Audio")
        audio_layout.addWidget(QtWidgets.QLabel("Audio Path"), 0, 0)
        audio_layout.addWidget(self.audio_path_edit, 0, 1)
        audio_layout.addWidget(self.audio_path_browse_button, 0, 2)
        audio_layout.addWidget(QtWidgets.QLabel("Audio Downsample"), 1, 0)
        audio_layout.addWidget(self.audio_downsample_spin, 1, 1)
        audio_layout.addWidget(self.play_audio_button, 2, 0)
        audio_layout.addWidget(self.stop_audio_button, 2, 1)
        audio_layout.addWidget(self.replay_audio_button, 2, 2)
        audio_layout.addWidget(self.export_audio_button, 3, 0, 1, 3)

        control_tabs = QtWidgets.QTabWidget()
        control_tabs.addTab(filter_group, "Display Controls")
        control_tabs.addTab(feature_group, "Short-Time Feature")
        control_tabs.addTab(audio_group, "Audio")

        control_layout.addWidget(directory_group)
        control_layout.addWidget(sort_group, stretch=1)
        control_layout.addWidget(control_tabs)
        control_layout.addStretch(1)

        right_panel = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        axis = AbsoluteTimeAxis("bottom")
        feature_axis = AbsoluteTimeAxis("bottom")
        self.time_plot = TimePlotWidget(axisItems={"bottom": axis})
        self.feature_plot = pg.PlotWidget(axisItems={"bottom": feature_axis})
        time_panel = QtWidgets.QWidget()
        time_layout = QtWidgets.QHBoxLayout(time_panel)
        time_layout.setContentsMargins(0, 0, 0, 0)
        time_layout.setSpacing(0)
        time_column_layout = QtWidgets.QVBoxLayout()
        time_column_layout.setContentsMargins(0, 0, 0, 0)
        time_column_layout.setSpacing(6)
        self.zoom_mode_button = QtWidgets.QPushButton("Zoom Mode")
        self.zoom_mode_button.setCheckable(True)
        self.zoom_mode_button.setChecked(UI_DEFAULTS.view.zoom_mode_checked)
        self.psd_mode_button = QtWidgets.QPushButton("Window PSD Mode")
        self.psd_mode_button.setCheckable(True)
        self.back_view_button = QtWidgets.QPushButton("Back View")
        self.back_view_button.setEnabled(False)
        self.fixed_psd_button = QtWidgets.QPushButton("PSD WINDOWS")
        self.fixed_psd_button.setCheckable(True)
        self.reset_view_button = QtWidgets.QPushButton("Reset View")
        self.zoom_out_button = QtWidgets.QPushButton("Zoom Out 2x")
        self.mode_button_group = QtWidgets.QButtonGroup(self)
        self.mode_button_group.setExclusive(True)
        self.mode_button_group.addButton(self.zoom_mode_button)
        self.mode_button_group.addButton(self.psd_mode_button)
        button_font = QtGui.QFont("Times New Roman", 12)
        for button in (
            self.zoom_mode_button,
            self.psd_mode_button,
            self.back_view_button,
            self.fixed_psd_button,
            self.zoom_out_button,
            self.reset_view_button,
        ):
            button.setFont(button_font)
            button.setMinimumHeight(34)
            button.setMinimumWidth(90)
            button.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.setSpacing(3)
        mode_row.addWidget(self.zoom_mode_button)
        mode_row.addWidget(self.psd_mode_button)
        mode_row.addWidget(self.back_view_button)
        mode_row.addWidget(self.fixed_psd_button)
        mode_row.addWidget(self.zoom_out_button)
        mode_row.addWidget(self.reset_view_button)
        self.visible_length_label = QtWidgets.QLabel("Visible: 0.000 s")
        self.window_length_label = QtWidgets.QLabel("Window: 0.000 s")
        self.visible_window_spin = QtWidgets.QDoubleSpinBox()
        self.visible_window_spin.setDecimals(3)
        self.visible_window_spin.setRange(0.001, 1e6)
        self.visible_window_spin.setSingleStep(0.001)
        self.visible_window_spin.setValue(UI_DEFAULTS.view.visible_window_seconds)
        self.apply_visible_window_button = QtWidgets.QPushButton("Apply Visible Window")
        self.feature_plot_mode_combo = QtWidgets.QComboBox()
        self.feature_plot_mode_combo.addItem("SVM Prediction", FEATURE_MODE_SVM)
        self.feature_plot_mode_combo.addItem("Short-Time Energy", FEATURE_MODE_ENERGY)
        info_font = QtGui.QFont("Times New Roman", 11)
        self.visible_length_label.setFont(info_font)
        self.window_length_label.setFont(info_font)
        self.time_scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        self.time_scrollbar.setEnabled(False)
        self.time_scrollbar.setSingleStep(1)
        self.time_scrollbar.setPageStep(1)
        mode_row.addSpacing(4)
        mode_row.addWidget(self.visible_length_label)
        mode_row.addSpacing(8)
        mode_row.addWidget(self.window_length_label)
        mode_row.addSpacing(8)
        mode_row.addWidget(QtWidgets.QLabel("Visible Window (s)"))
        mode_row.addWidget(self.visible_window_spin)
        mode_row.addWidget(self.apply_visible_window_button)
        mode_row.addSpacing(14)
        mode_row.addWidget(QtWidgets.QLabel("Plot 2"))
        mode_row.addWidget(self.feature_plot_mode_combo)
        mode_row.addStretch(1)
        self.psd_plot = pg.PlotWidget()
        tf_axis = AbsoluteTimeAxis("bottom")
        tf_freq_axis = LogFrequencyAxis("left")
        self.tf_plot = pg.PlotWidget(axisItems={"bottom": tf_axis, "left": tf_freq_axis})
        configure_plot_widget(self.time_plot, "Phase (rad)", "Time")
        configure_plot_widget(self.feature_plot, "SVM Prediction", "Time")
        configure_plot_widget(self.psd_plot, "PSD (dB rad^2/Hz)", "Frequency (Hz)")
        configure_plot_widget(self.tf_plot, "Frequency (Hz, log)", "Time")
        aligned_left_axis_width = 90
        self.time_plot.getPlotItem().getAxis("left").setWidth(aligned_left_axis_width)
        self.tf_plot.getPlotItem().getAxis("left").setWidth(aligned_left_axis_width)
        self.psd_plot.setLogMode(x=True, y=False)
        self.feature_plot.setXLink(self.time_plot)
        self.tf_plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.tf_plot.getViewBox().setMouseEnabled(x=True, y=True)
        self.time_curve = self.time_plot.plot(pen=make_pen("#CC2222", 1))
        self.time_curve.setClipToView(True)
        self.time_curve.setDownsampling(auto=True, method="peak")
        self.time_curve.setSkipFiniteCheck(True)
        self.feature_curve = self.feature_plot.plot(pen=make_pen("#2266AA", 2))
        self.feature_curve.setClipToView(True)
        self.feature_curve.setSkipFiniteCheck(True)
        self.psd_curve = self.psd_plot.plot(pen=make_pen("#AA3333", 1))
        self.psd_curve.setSkipFiniteCheck(True)
        self.tf_image_item = pg.ImageItem(axisOrder="row-major")
        self.tf_plot.addItem(self.tf_image_item)
        self.tf_histogram = pg.HistogramLUTWidget()
        self.tf_histogram.setImageItem(self.tf_image_item)
        self.tf_histogram.setBackground("#FFFFFF")
        self.tf_histogram.setStyleSheet("background: #FFFFFF;")
        self.tf_histogram.item.vb.setBackgroundColor("#FFFFFF")
        self.tf_histogram.setMinimumWidth(120)
        self.tf_histogram.setMaximumWidth(120)
        time_column_layout.addWidget(self.time_plot, stretch=1)
        time_column_layout.addWidget(self.time_scrollbar)
        time_column_layout.addLayout(mode_row)
        time_layout.addLayout(time_column_layout, stretch=1)
        self.time_right_spacer = QtWidgets.QWidget()
        self.time_right_spacer.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        self.time_right_spacer.setMinimumWidth(0)
        self.time_right_spacer.setMaximumWidth(0)
        time_layout.addWidget(self.time_right_spacer)
        curve_tab = QtWidgets.QWidget()
        curve_tab_layout = QtWidgets.QVBoxLayout(curve_tab)
        curve_tab_layout.setContentsMargins(0, 0, 0, 0)
        curve_tab_layout.setSpacing(0)
        curve_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        curve_splitter.addWidget(self.feature_plot)
        curve_splitter.addWidget(self.psd_plot)
        curve_splitter.setSizes([220, 320])
        curve_tab_layout.addWidget(curve_splitter)
        tf_tab = QtWidgets.QWidget()
        tf_tab_layout = QtWidgets.QHBoxLayout(tf_tab)
        tf_tab_layout.setContentsMargins(0, 0, 0, 0)
        tf_tab_layout.setSpacing(6)
        tf_tab_layout.addWidget(self.tf_plot, stretch=1)
        tf_tab_layout.addWidget(self.tf_histogram)
        self.analysis_tabs = QtWidgets.QTabWidget()
        self.analysis_tabs.setObjectName("analysisTabs")
        self.analysis_tabs.addTab(curve_tab, "1D Curve")
        self.analysis_tabs.addTab(tf_tab, "t-f Plot")
        self._update_time_tf_alignment_for_tab(self.analysis_tabs.currentIndex())
        right_panel.addWidget(time_panel)
        right_panel.addWidget(self.analysis_tabs)
        right_panel.setSizes([360, 540])
        self.tf_color_min_spin.setEnabled(False)
        self.tf_color_max_spin.setEnabled(False)

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_splitter.addWidget(control_scroll)
        main_splitter.addWidget(right_panel)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([440, 1060])
        main_layout.addWidget(main_splitter, stretch=1)

        self.statusBar().showMessage("Ready.")
        self._apply_fonts()
        self._apply_theme()
        self._update_interaction_mode()
        self._update_feature_plot_style()
        self._handle_tf_color_auto_toggled(self.tf_color_auto_checkbox.isChecked())
        self._apply_time_frequency_colormap()

    def _apply_fonts(self) -> None:
        self.setFont(QtGui.QFont("SimSun", 10))

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #F5F7FA;
                color: #1F2937;
            }
            QLabel {
                color: #1F2937;
            }
            QStatusBar {
                background: #EEF2F7;
                color: #475569;
                border-top: 1px solid #D7DEE8;
            }
            QGroupBox {
                background: #FFFFFF;
                border: 1px solid #94A3B8;
                border-radius: 8px;
                margin-top: 10px;
                font-weight: 600;
                color: #1F2937;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                background: #FFFFFF;
                color: #334155;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget {
                background: #FFFFFF;
                border: 1px solid #C8D2DF;
                border-radius: 6px;
                padding: 4px 6px;
                color: #1F2937;
                selection-background-color: #2F6FAE;
                selection-color: #FFFFFF;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QListWidget:focus {
                border: 1px solid #2F6FAE;
            }
            QPushButton {
                background: #E8EDF4;
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                color: #1F2937;
                padding: 5px 10px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #DFE7F1;
                border: 1px solid #B8C4D6;
            }
            QPushButton:pressed {
                background: #2F6FAE;
                border: 1px solid #255C91;
                color: #FFFFFF;
            }
            QPushButton:checked {
                background: #2F6FAE;
                border: 1px solid #255C91;
                color: #FFFFFF;
            }
            QPushButton:disabled {
                background: #EEF2F7;
                border: 1px solid #D7DEE8;
                color: #94A3B8;
            }
            QTabWidget::pane {
                border: 1px solid #94A3B8;
                border-radius: 8px;
                background: #FFFFFF;
                top: -1px;
            }
            QTabBar::tab {
                background: #E9EEF5;
                color: #475569;
                border: 1px solid #D7DEE8;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 6px 12px;
                margin-right: 2px;
                font-weight: 700;
                min-width: 170px;
            }
            QTabBar::tab:selected {
                background: #FFFFFF;
                color: #1F2937;
            }
            QTabWidget#analysisTabs QTabBar::tab {
                background: #DDE6F3;
                color: #334155;
                border: 1px solid #A7B7CF;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 7px 14px;
                margin-right: 2px;
                min-width: 130px;
                font-weight: 700;
            }
            QTabWidget#analysisTabs QTabBar::tab:selected {
                background: #1E40AF;
                color: #FFFFFF;
                border: 1px solid #1E3A8A;
            }
            QScrollArea {
                border: none;
                background: #F5F7FA;
            }
            QSplitter::handle {
                background: #D7DEE8;
            }
            QSplitter::handle:hover {
                background: #BFCADD;
            }
            QScrollBar:horizontal {
                background: #E9EEF5;
                height: 12px;
                border-radius: 6px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: #A8B4C6;
                border-radius: 6px;
                min-width: 24px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #8E9CB2;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            """
        )

    def _bind_events(self) -> None:
        self.browse_button.clicked.connect(self._choose_directory)
        self.export_browse_button.clicked.connect(self._choose_export_directory)
        self.audio_path_browse_button.clicked.connect(self._choose_audio_path)
        self.audio_path_edit.textEdited.connect(self._handle_audio_path_edited)
        self.refresh_button.clicked.connect(self._refresh_file_list)
        self.export_visible_button.clicked.connect(self._export_visible_raw_data)
        self.play_audio_button.clicked.connect(self._play_visible_audio)
        self.stop_audio_button.clicked.connect(self._stop_audio_playback)
        self.replay_audio_button.clicked.connect(self._replay_visible_audio)
        self.export_audio_button.clicked.connect(self._export_visible_audio)
        self.sort_field_combo.currentIndexChanged.connect(self._refresh_file_list)
        self.sort_order_combo.currentIndexChanged.connect(self._refresh_file_list)
        self.file_list.currentRowChanged.connect(self._handle_file_selection)
        self.home_button.clicked.connect(lambda: self._change_page(0))
        self.prev_button.clicked.connect(lambda: self._change_page(self._page_index - 1))
        self.next_button.clicked.connect(lambda: self._change_page(self._page_index + 1))
        self.end_button.clicked.connect(self._go_to_last_page)
        self.jump_button.clicked.connect(self._jump_to_page)
        self.threshold_filter_button.clicked.connect(self._apply_amplitude_threshold_filter)
        self.apply_filter_button.clicked.connect(self._rebuild_time_plot)
        self.apply_y_range_button.clicked.connect(self._apply_y_range)
        self.apply_tf_button.clicked.connect(self._apply_time_frequency_params)
        self.tf_color_auto_checkbox.toggled.connect(self._handle_tf_color_auto_toggled)
        self.tf_color_min_spin.valueChanged.connect(self._handle_tf_color_min_changed)
        self.analysis_tabs.currentChanged.connect(self._update_time_tf_alignment_for_tab)
        self.feature_apply_y_range_button.clicked.connect(self._apply_feature_y_range)
        self.feature_apply_button.clicked.connect(self._rebuild_short_time_feature_plot)
        self.apply_visible_window_button.clicked.connect(self._apply_visible_window_duration)
        self.feature_plot_mode_combo.currentIndexChanged.connect(self._handle_feature_mode_changed)
        self.zoom_mode_button.clicked.connect(
            lambda checked: checked and self._set_interaction_mode(InteractionMode.ZOOM)
        )
        self.psd_mode_button.clicked.connect(
            lambda checked: checked and self._set_interaction_mode(InteractionMode.WINDOW_PSD)
        )
        self.back_view_button.clicked.connect(self._restore_previous_view)
        self.zoom_out_button.clicked.connect(self._zoom_out_time_plot)
        self.reset_view_button.clicked.connect(self._reset_time_plot)
        self.fixed_psd_button.toggled.connect(self._toggle_fixed_psd_window)
        self.time_scrollbar.valueChanged.connect(self._handle_time_scrollbar_change)
        self.time_plot.windowSelected.connect(self._update_psd_from_selection)
        self.time_plot.clearSelectionRequested.connect(self._clear_selection)
        self.time_plot.getViewBox().sigRangeChanged.connect(self._record_view_history)
        self.time_plot.getViewBox().sigRangeChanged.connect(self._update_visible_length_label)
        self.time_plot.getViewBox().sigRangeChanged.connect(self._handle_time_view_changed)
        self.time_plot.getViewBox().sigXRangeChanged.connect(self._sync_tf_x_from_time)
        self.tf_plot.getViewBox().sigXRangeChanged.connect(self._sync_time_x_from_tf)
        self._bind_horizontal_scroll_shortcuts()

    def _handle_tf_color_auto_toggled(self, checked: bool) -> None:
        manual_enabled = not bool(checked)
        self.tf_color_min_spin.setEnabled(manual_enabled)
        self.tf_color_max_spin.setEnabled(manual_enabled)
        self._apply_time_frequency_color_levels()

    def _handle_tf_color_min_changed(self, value: float) -> None:
        if self._updating_tf_color_spins:
            return
        self._tf_color_min_user_override = abs(float(value) - self._tf_default_color_min) > 1e-9
        if self.tf_color_auto_checkbox.isChecked():
            self._apply_time_frequency_color_levels()

    def _update_time_tf_alignment_for_tab(self, index: int) -> None:
        target_width = self._tf_side_panel_width if int(index) == 1 else 0
        self.time_right_spacer.setMinimumWidth(target_width)
        self.time_right_spacer.setMaximumWidth(target_width)

    def _bind_horizontal_scroll_shortcuts(self) -> None:
        for key, direction in (
            (QtCore.Qt.Key_Left, -1),
            (QtCore.Qt.Key_Right, 1),
        ):
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            shortcut.setAutoRepeat(True)
            shortcut.activated.connect(
                lambda direction=direction: self._scroll_time_plot_by_step(direction)
            )
            self._scroll_shortcuts.append(shortcut)

    def _choose_directory(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Data Directory",
            self.directory_edit.text(),
        )
        if directory:
            self.directory_edit.setText(directory)
            self._page_index = 0
            self._refresh_file_list()

    def _choose_export_directory(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory",
            self.export_directory_edit.text(),
        )
        if directory:
            self.export_directory_edit.setText(directory)

    def _choose_audio_path(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select Audio File",
            self.audio_path_edit.text().strip(),
            "WAV Audio (*.wav)",
        )
        if path:
            self._audio_path_auto_managed = False
            self.audio_path_edit.setText(path)

    def _handle_audio_path_edited(self, _text: str) -> None:
        self._audio_path_auto_managed = False

    def _default_audio_path(self) -> Path:
        if self._current_waveform is None:
            timestamp = "000000000000"
            return Path.cwd() / "exports" / f"{timestamp}.wav"

        x_range, _ = self.time_plot.getViewBox().viewRange()
        start_x, _end_x = self._normalized_x_range((float(x_range[0]), float(x_range[1])))
        start_index = max(0, int(np.floor(start_x)))
        start_time = self._current_waveform.start_time + timedelta(
            seconds=start_index / self._current_waveform.sample_rate
        )
        timestamp = start_time.strftime("%Y%m%d%H%M")
        return Path.cwd() / "exports" / f"{timestamp}.wav"

    def _export_visible_raw_data(self) -> None:
        if self._current_waveform is None:
            self.statusBar().showMessage('No waveform is loaded.')
            return

        export_directory_text = self.export_directory_edit.text().strip()
        if not export_directory_text:
            self.statusBar().showMessage('Please set an export directory first.')
            return
        export_directory = Path(export_directory_text)

        x_range, _ = self.time_plot.getViewBox().viewRange()
        start_x, end_x = self._normalized_x_range((float(x_range[0]), float(x_range[1])))
        start_index = max(0, int(np.floor(start_x)))
        end_index = min(self._current_waveform.phase_data.size, int(np.ceil(end_x)))
        if end_index <= start_index:
            end_index = min(self._current_waveform.phase_data.size, start_index + 1)
        if end_index <= start_index:
            self.statusBar().showMessage('Visible window is empty.')
            return

        segment = self._current_waveform.phase_data[start_index:end_index]
        segment_start_time = self._current_waveform.start_time + timedelta(
            seconds=start_index / self._current_waveform.sample_rate
        )
        filename = build_export_tdms_name(segment_start_time, self._current_waveform.sample_rate)
        destination = export_directory / filename

        try:
            save_tdms_waveform(
                destination,
                segment,
                self._current_waveform.sample_rate,
                segment_start_time,
            )
        except Exception as exc:
            self.statusBar().showMessage(f'Export failed: {exc}')
            return

        self.statusBar().showMessage(
            f'Exported visible raw data to {destination}'
        )

    def _get_visible_display_segment(self) -> Optional[tuple[np.ndarray, float, int, int]]:
        if self._current_waveform is None or self._current_display_values.size == 0:
            self.statusBar().showMessage('No waveform is loaded.')
            return None

        x_range, _ = self.time_plot.getViewBox().viewRange()
        start_x, end_x = self._normalized_x_range((float(x_range[0]), float(x_range[1])))
        start_index = max(0, int(np.floor(start_x)))
        end_index = min(self._current_display_values.size, int(np.ceil(end_x)))
        if end_index <= start_index:
            end_index = min(self._current_display_values.size, start_index + 1)
        if end_index <= start_index:
            self.statusBar().showMessage('Visible waveform window is empty.')
            return None

        segment = np.asarray(self._current_display_values[start_index:end_index], dtype=np.float64)
        return segment, float(self._current_waveform.sample_rate), start_index, end_index

    def _build_visible_audio_pcm(self) -> Optional[tuple[np.ndarray, int, int, int]]:
        segment_info = self._get_visible_display_segment()
        if segment_info is None:
            return None

        segment, sample_rate, start_index, end_index = segment_info
        downsample_factor = int(self.audio_downsample_spin.value())
        try:
            pcm16, audio_sample_rate = prepare_audio_waveform(
                segment,
                sample_rate,
                downsample_factor,
            )
        except Exception as exc:
            self.statusBar().showMessage(f'Failed to build audio: {exc}')
            return None
        return pcm16, audio_sample_rate, start_index, end_index

    def _clear_audio_temp_path(self) -> None:
        if self._audio_temp_path is not None and self._audio_temp_path.exists():
            try:
                self._audio_temp_path.unlink()
            except OSError:
                pass
        self._audio_temp_path = None

    def _prepare_visible_audio_media(self) -> Optional[tuple[int, int, int]]:
        audio_info = self._build_visible_audio_pcm()
        if audio_info is None:
            return None

        pcm16, audio_sample_rate, start_index, end_index = audio_info
        try:
            self._audio_player.stop()
            self._clear_audio_temp_path()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                temp_path = Path(handle.name)
            save_wav_waveform(temp_path, pcm16, audio_sample_rate)
        except Exception as exc:
            self.statusBar().showMessage(f'Failed to prepare playback audio: {exc}')
            return None

        self._audio_temp_path = temp_path
        self._audio_player.setMedia(
            QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(str(temp_path)))
        )
        return audio_sample_rate, start_index, end_index

    def _play_visible_audio(self) -> None:
        state = self._audio_player.state()
        if state == QtMultimedia.QMediaPlayer.PausedState:
            self._audio_player.play()
            self.statusBar().showMessage('Audio playback resumed.')
            return
        if self._audio_temp_path is None or not self._audio_temp_path.exists():
            media_info = self._prepare_visible_audio_media()
            if media_info is None:
                return
            audio_sample_rate, start_index, end_index = media_info
            duration_seconds = max(0.0, self._audio_player.duration() / 1000.0)
            self._audio_player.play()
            self.statusBar().showMessage(
                f'Playing visible audio: samples {start_index} to {end_index}, '
                f'{audio_sample_rate} Hz, {duration_seconds:.3f} s.'
            )
            return
        self._audio_player.play()
        duration_seconds = max(0.0, self._audio_player.duration() / 1000.0)
        self.statusBar().showMessage(
            f'Audio playback started: {duration_seconds:.3f} s.'
        )

    def _stop_audio_playback(self) -> None:
        if self._audio_player.state() == QtMultimedia.QMediaPlayer.StoppedState:
            self.statusBar().showMessage('Audio is not playing.')
            return
        self._audio_player.pause()
        self.statusBar().showMessage('Audio playback stopped.')

    def _replay_visible_audio(self) -> None:
        if self._audio_temp_path is None or not self._audio_temp_path.exists():
            media_info = self._prepare_visible_audio_media()
            if media_info is None:
                return
            audio_sample_rate, start_index, end_index = media_info
        else:
            audio_sample_rate = 0
            start_index = 0
            end_index = 0
        self._audio_player.setPosition(0)
        self._audio_player.play()
        duration_seconds = max(0.0, self._audio_player.duration() / 1000.0)
        if audio_sample_rate > 0:
            self.statusBar().showMessage(
                f'Replaying visible audio: samples {start_index} to {end_index}, '
                f'{audio_sample_rate} Hz, {duration_seconds:.3f} s.'
            )
        else:
            self.statusBar().showMessage(f'Audio replayed from start: {duration_seconds:.3f} s.')

    def _export_visible_audio(self) -> None:
        if self._current_waveform is None:
            self.statusBar().showMessage('No waveform is loaded.')
            return

        audio_path_text = self.audio_path_edit.text().strip()
        if not audio_path_text:
            self._audio_path_auto_managed = True
            audio_path = self._default_audio_path()
            self.audio_path_edit.setText(str(audio_path))
        else:
            audio_path = Path(audio_path_text)

        audio_info = self._build_visible_audio_pcm()
        if audio_info is None:
            return

        pcm16, audio_sample_rate, start_index, end_index = audio_info
        try:
            destination = save_wav_waveform(audio_path, pcm16, audio_sample_rate)
        except Exception as exc:
            self.statusBar().showMessage(f'Audio export failed: {exc}')
            return

        duration_seconds = pcm16.size / max(audio_sample_rate, 1)
        self.statusBar().showMessage(
            f'Exported visible audio to {destination} '
            f'({audio_sample_rate} Hz, {duration_seconds:.3f} s, source samples {start_index}-{end_index}).'
        )

    def _refresh_file_list(self) -> None:
        directory = Path(self.directory_edit.text().strip())
        sort_field = self.sort_field_combo.currentData()
        ascending = bool(self.sort_order_combo.currentData())

        try:
            self._source_files = list_data_files(directory, sort_field=sort_field, ascending=ascending)
            self._all_files = list(self._source_files)
            self.statusBar().showMessage(f"Loaded directory: {directory}")
        except Exception as exc:
            self._source_files = []
            self._all_files = []
            self.file_list.clear()
            self.page_info_label.setText("Page 0 / 0 | Total 0")
            self.page_jump_spin.setMaximum(1)
            self.page_jump_spin.setValue(1)
            self.statusBar().showMessage(str(exc))
            return

        self._change_page(self._page_index)

    def _change_page(self, page_index: int) -> None:
        page = paginate_files(self._all_files, page_index=page_index, page_size=PAGE_SIZE)
        self._page_index = page.page_index
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for record in page.items:
            list_item = QtWidgets.QListWidgetItem(record.name)
            list_item.setData(QtCore.Qt.UserRole, record.path)
            self.file_list.addItem(list_item)
        self.file_list.blockSignals(False)

        current_page = page.page_index + 1 if page.total_count else 0
        display_page_count = page.page_count if page.total_count else 0
        self.page_info_label.setText(
            f"Page {current_page} / {display_page_count} | Total {page.total_count}"
        )
        self.page_jump_spin.setMaximum(max(1, display_page_count))
        self.page_jump_spin.setValue(max(1, current_page))

        self.home_button.setEnabled(page.page_index > 0)
        self.prev_button.setEnabled(page.page_index > 0)
        self.next_button.setEnabled(page.page_index < page.page_count - 1 and page.total_count > 0)
        self.end_button.setEnabled(page.page_index < page.page_count - 1 and page.total_count > 0)
        self.jump_button.setEnabled(page.total_count > 0)

        if self.file_list.count() > 0:
            self.file_list.setCurrentRow(0)
        else:
            self._current_waveform = None
            self._current_display_values = np.array([], dtype=np.float64)
            self._audio_player.stop()
            self._clear_audio_temp_path()
            self.time_curve.setData([])
            self._clear_short_time_feature_plot()
            self.psd_curve.setData([], [])
            self._clear_time_frequency_plot()
            self._set_fixed_psd_enabled(False)
            self.time_plot.clear_selection_region()
            self.visible_length_label.setText("Visible: 0.000 s")
            self.window_length_label.setText("Window: 0.000 s")
            self._audio_path_auto_managed = True
            self.audio_path_edit.setText(str(self._default_audio_path()))
            self._update_time_scrollbar()

    def _go_to_last_page(self) -> None:
        page_count = paginate_files(self._all_files, 0, PAGE_SIZE).page_count
        self._change_page(max(0, page_count - 1))

    def _jump_to_page(self) -> None:
        if not self._all_files:
            return
        self._change_page(int(self.page_jump_spin.value()) - 1)

    def _handle_file_selection(self, row: int) -> None:
        if row < 0:
            return
        item = self.file_list.item(row)
        if item is None:
            return

        path = item.data(QtCore.Qt.UserRole)
        if not path:
            return

        self._start_waveform_load(Path(path))

    def _build_display_values(
        self, waveform: Optional[LoadedWaveform], *, strict_validation: bool = False
    ) -> Optional[np.ndarray]:
        if waveform is None:
            return None

        values = waveform.phase_data
        enabled = self.filter_enabled_checkbox.isChecked()
        mode = self.filter_mode_combo.currentData()
        low_cut = float(self.low_cut_spin.value())
        high_cut = float(self.high_cut_spin.value())
        valid, message = validate_filter(
            enabled=enabled,
            mode=mode,
            sample_rate=waveform.sample_rate,
            low_cut_hz=low_cut,
            high_cut_hz=high_cut,
        )
        if not valid:
            self.statusBar().showMessage(message)
            if strict_validation:
                return None
            return values

        return apply_display_filter(
            values=values,
            sample_rate=waveform.sample_rate,
            enabled=enabled,
            mode=mode,
            low_cut_hz=low_cut,
            high_cut_hz=high_cut,
        )

    def _get_display_values(self) -> Optional[np.ndarray]:
        return self._build_display_values(self._current_waveform)

    def _apply_amplitude_threshold_filter(self) -> None:
        if not self._source_files:
            self.statusBar().showMessage("No files available for threshold filtering.")
            return

        threshold = float(self.amplitude_threshold_spin.value())
        progress = QtWidgets.QProgressDialog(
            "Filtering files by amplitude threshold...",
            None,
            0,
            len(self._source_files),
            self,
        )
        progress.setWindowTitle("Threshold Filter")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.show()

        kept_files: list[FileRecord] = []
        failed_files = 0
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for index, record in enumerate(self._source_files, start=1):
                progress.setLabelText(f"Filtering {record.name} ({index}/{len(self._source_files)})")
                progress.setValue(index - 1)
                QtWidgets.QApplication.processEvents()

                try:
                    waveform = load_waveform(record.path)
                    display_values = self._build_display_values(waveform, strict_validation=True)
                    if display_values is None:
                        failed_files += 1
                        continue
                    peak_amplitude = float(np.max(np.abs(display_values))) if display_values.size else 0.0
                    if peak_amplitude >= threshold:
                        kept_files.append(record)
                except Exception:
                    failed_files += 1

            self._all_files = kept_files
            self._page_index = 0
            progress.setValue(len(self._source_files))
            self._change_page(0)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self.statusBar().showMessage(
            f"Threshold filter finished. Kept {len(self._all_files)} / {len(self._source_files)} files"
            f" with peak amplitude >= {threshold:.6f}."
            + (f" Failed to process {failed_files} files." if failed_files else "")
        )

    def _start_waveform_load(self, path: Path) -> None:
        self._load_task_id += 1
        task_id = self._load_task_id
        self._prediction_task_id += 1
        self._current_waveform = None
        self._current_display_values = np.array([], dtype=np.float64)
        self._audio_player.stop()
        self._clear_audio_temp_path()
        self.time_curve.setData([])
        self._clear_short_time_feature_plot()
        self.psd_curve.setData([], [])
        self._clear_time_frequency_plot()
        self._set_fixed_psd_enabled(False)
        self.time_plot.clear_selection_region()
        self.visible_length_label.setText("Visible: 0.000 s")
        self.window_length_label.setText("Window: 0.000 s")
        self._update_time_scrollbar()
        self.statusBar().showMessage(f"Loading {path.name}...")

        thread = QtCore.QThread(self)
        worker = LoadWaveformWorker(task_id, path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._handle_waveform_loaded)
        worker.failed.connect(self._handle_waveform_load_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._handle_load_thread_finished)
        self._load_thread = thread
        self._load_worker = worker
        thread.start()

    @QtCore.pyqtSlot()
    def _handle_load_thread_finished(self) -> None:
        self._load_thread = None
        self._load_worker = None

    @QtCore.pyqtSlot(int, object)
    def _handle_waveform_loaded(self, task_id: int, waveform: LoadedWaveform) -> None:
        if task_id != self._load_task_id:
            return
        self._current_waveform = waveform
        self._rebuild_time_plot()
        warning = waveform.data_info_warning
        if warning:
            self.statusBar().showMessage(f"{waveform.path.name}: {warning}")
        else:
            self.statusBar().showMessage(f"Opened {waveform.path.name}")

    @QtCore.pyqtSlot(int, str)
    def _handle_waveform_load_failed(self, task_id: int, message: str) -> None:
        if task_id != self._load_task_id:
            return
        self._current_waveform = None
        self.statusBar().showMessage(f"Failed to open file: {message}")

    def _start_short_time_feature_prediction(self) -> None:
        if self._current_waveform is None:
            self._clear_short_time_feature_plot()
            return

        self._prediction_task_id += 1
        task_id = self._prediction_task_id
        self._clear_short_time_feature_plot()
        self.statusBar().showMessage(f"Running SVM prediction for {self._current_waveform.path.name}...")

        thread = QtCore.QThread(self)
        worker = SVMPredictionWorker(
            task_id,
            self._svm_model_directory,
            self._current_waveform.phase_data,
            float(self._current_waveform.sample_rate),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._handle_prediction_finished)
        worker.failed.connect(self._handle_prediction_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._handle_prediction_thread_finished)
        self._prediction_thread = thread
        self._prediction_worker = worker
        thread.start()

    @QtCore.pyqtSlot()
    def _handle_prediction_thread_finished(self) -> None:
        self._prediction_thread = None
        self._prediction_worker = None

    @QtCore.pyqtSlot(int, object, object)
    def _handle_prediction_finished(self, task_id: int, centers: np.ndarray, predictions: np.ndarray) -> None:
        if task_id != self._prediction_task_id or self._current_feature_mode() != FEATURE_MODE_SVM:
            return
        if centers.size == 0:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage('SVM prediction produced no valid windows.')
            return

        self.feature_curve.setData(centers, predictions)
        self._apply_feature_y_range()
        self.statusBar().showMessage(f'SVM prediction updated for {centers.size} windows.')

    @QtCore.pyqtSlot(int, str)
    def _handle_prediction_failed(self, task_id: int, message: str) -> None:
        if task_id != self._prediction_task_id or self._current_feature_mode() != FEATURE_MODE_SVM:
            return
        self._clear_short_time_feature_plot()
        self.statusBar().showMessage(f'Failed to compute SVM prediction: {message}')

    def _rebuild_time_plot(self) -> None:
        if self._current_waveform is None:
            return

        values = self._get_display_values()
        if values is None:
            return

        self._current_display_values = values
        self.time_curve.setData(values)
        for plot_widget in (self.time_plot, self.feature_plot, self.tf_plot):
            bottom_axis = plot_widget.getPlotItem().getAxis("bottom")
            if isinstance(bottom_axis, AbsoluteTimeAxis):
                bottom_axis.set_context(
                    start_time=self._current_waveform.start_time,
                    sample_rate=self._current_waveform.sample_rate,
                )
        self.time_plot.set_data_context(
            data_length=len(values),
            sample_rate=self._current_waveform.sample_rate,
            min_window_seconds=0.001,
        )
        self._set_fixed_psd_enabled(False)
        self.time_plot.clear_selection_region()
        self._clear_short_time_feature_plot()
        self.psd_curve.setData([], [])
        self._clear_time_frequency_plot()
        self.window_length_label.setText("Window: 0.000 s")
        self._clear_view_history()
        self._apply_view_state(
            ((0.0, float(max(1, len(self._current_display_values) - 1))), self._default_y_range())
        )
        self._refresh_default_audio_path()
        self._rebuild_short_time_feature_plot()
        self._rebuild_time_frequency_plot()

    def _apply_y_range(self) -> None:
        if self._current_waveform is None:
            self._apply_psd_y_range()
            self._apply_time_frequency_y_range()
            self._apply_time_frequency_color_levels()
            return

        y_min = float(self.y_min_spin.value())
        y_max = float(self.y_max_spin.value())
        view_box = self.time_plot.getViewBox()
        if y_min == 0.0 and y_max == 0.0:
            view_box.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        elif y_min >= y_max:
            view_box.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            self.statusBar().showMessage("Invalid Y range. Switched to auto range.")
        else:
            view_box.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
            view_box.setYRange(y_min, y_max, padding=0.0)
        self._apply_psd_y_range()
        self._apply_time_frequency_y_range()
        self._apply_time_frequency_color_levels()

    def _apply_time_frequency_params(self) -> None:
        self._rebuild_time_frequency_plot()

    def _current_tf_mode(self) -> str:
        return str(self.tf_mode_combo.currentData())

    def _current_tf_scale(self) -> str:
        return str(self.tf_value_scale_combo.currentData())

    def _current_tf_colormap_name(self) -> str:
        return self.tf_colormap_combo.currentText().strip() or "jet"

    def _clear_time_frequency_plot(self) -> None:
        self._tf_freq_hz = np.array([], dtype=np.float64)
        self._tf_time_centers = np.array([], dtype=np.float64)
        self._tf_base_values = np.array([], dtype=np.float64)
        self._tf_log_freq_bounds = None
        self.tf_image_item.setImage(np.empty((0, 0), dtype=np.float64), autoLevels=False)

    def _rebuild_time_frequency_plot(self) -> None:
        if self._current_waveform is None:
            self._clear_time_frequency_plot()
            return
        source_values = np.asarray(self._current_display_values, dtype=np.float64)
        if source_values.size == 0:
            refreshed = self._get_display_values()
            if refreshed is None:
                self._clear_time_frequency_plot()
                return
            source_values = np.asarray(refreshed, dtype=np.float64)
        if source_values.size < 8:
            self._clear_time_frequency_plot()
            return

        window_seconds = float(self.tf_window_spin.value())
        overlap_ratio = float(self.tf_overlap_spin.value()) / 100.0
        if window_seconds <= 0.0:
            self._clear_time_frequency_plot()
            self.statusBar().showMessage("t-f window must be greater than 0 s.")
            return
        if overlap_ratio < 0.0 or overlap_ratio >= 1.0:
            self._clear_time_frequency_plot()
            self.statusBar().showMessage("t-f overlap must be in [0, 100).")
            return

        mode = self._current_tf_mode()
        try:
            freqs, centers, values_map = compute_time_frequency_map(
                source_values,
                float(self._current_waveform.sample_rate),
                window_seconds=window_seconds,
                overlap_ratio=overlap_ratio,
                spectrum_mode=mode,
            )
        except Exception as exc:
            self._clear_time_frequency_plot()
            self.statusBar().showMessage(f"Failed to compute t-f plot: {exc}")
            return

        if freqs.size == 0 or centers.size == 0 or values_map.size == 0:
            self._clear_time_frequency_plot()
            self.statusBar().showMessage("t-f parameters produced no valid windows.")
            return

        self._tf_freq_hz = np.asarray(freqs, dtype=np.float64)
        self._tf_time_centers = np.asarray(centers, dtype=np.float64)
        self._tf_base_values = np.asarray(values_map, dtype=np.float64)
        self._tf_base_mode = mode
        self._render_time_frequency_image()

    def _render_time_frequency_image(self) -> None:
        if (
            self._tf_freq_hz.size == 0
            or self._tf_time_centers.size == 0
            or self._tf_base_values.size == 0
        ):
            self._clear_time_frequency_plot()
            return

        valid = self._tf_freq_hz > 0.0
        if not np.any(valid):
            self._clear_time_frequency_plot()
            self.statusBar().showMessage("t-f plot requires positive frequencies.")
            return

        freqs = self._tf_freq_hz[valid]
        base_values = self._tf_base_values[valid, :]
        floor = np.finfo(np.float64).tiny
        if self._current_tf_scale() == TF_SCALE_LOG:
            if self._tf_base_mode == TF_MODE_PSD:
                display_values = 10.0 * np.log10(np.maximum(base_values, floor))
            else:
                display_values = 20.0 * np.log10(np.maximum(base_values, floor))
        else:
            display_values = np.asarray(base_values, dtype=np.float64)

        log_freq = np.log10(freqs)
        self._tf_log_freq_bounds = (float(log_freq[0]), float(log_freq[-1]))
        self.tf_image_item.setImage(display_values, autoLevels=False)
        self._apply_time_frequency_colormap()

        x0 = float(self._tf_time_centers[0])
        x1 = float(self._tf_time_centers[-1])
        dx = float(np.median(np.diff(self._tf_time_centers))) if self._tf_time_centers.size > 1 else 1.0
        dy = float(np.median(np.diff(log_freq))) if log_freq.size > 1 else 0.01
        width = max(dx, (x1 - x0) + dx)
        height = max(dy, (float(log_freq[-1]) - float(log_freq[0])) + dy)
        self.tf_image_item.setRect(
            QtCore.QRectF(
                x0 - 0.5 * dx,
                float(log_freq[0]) - 0.5 * dy,
                width,
                height,
            )
        )
        self.tf_plot.getPlotItem().setLimits(
            xMin=x0 - dx,
            xMax=x1 + dx,
            yMin=float(log_freq[0]) - dy,
            yMax=float(log_freq[-1]) + dy,
        )
        self._apply_time_frequency_y_range()
        self._apply_time_frequency_color_levels(display_values)
        self._sync_tf_x_from_time()
        mode_label = "PSD" if self._tf_base_mode == TF_MODE_PSD else "Amplitude"
        self.statusBar().showMessage(
            f"t-f plot updated: {mode_label}, {self._tf_time_centers.size} windows, {freqs.size} frequency bins."
        )

    def _apply_time_frequency_colormap(self) -> None:
        color_map = create_colormap(self._current_tf_colormap_name())
        self.tf_image_item.setColorMap(color_map)
        self.tf_histogram.item.gradient.setColorMap(color_map)

    def _apply_time_frequency_y_range(self) -> None:
        if self._tf_log_freq_bounds is None:
            return
        view_box = self.tf_plot.getViewBox()
        lower_bound, upper_bound = self._tf_log_freq_bounds
        y_min = float(self.tf_y_min_spin.value())
        y_max = float(self.tf_y_max_spin.value())
        if y_min == 0.0 and y_max == 0.0:
            view_box.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
            view_box.setYRange(lower_bound, upper_bound, padding=0.0)
            return
        if y_min <= 0.0 or y_max <= 0.0 or y_min >= y_max:
            self.statusBar().showMessage("Invalid t-f Y range. Kept previous range.")
            return
        log_min = max(np.log10(y_min), lower_bound)
        log_max = min(np.log10(y_max), upper_bound)
        if log_min >= log_max:
            self.statusBar().showMessage("t-f Y range is outside available frequencies.")
            return
        view_box.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
        view_box.setYRange(float(log_min), float(log_max), padding=0.0)

    def _apply_time_frequency_color_levels(self, values: Optional[np.ndarray] = None) -> None:
        if values is None:
            if self._tf_base_values.size == 0:
                return
            valid = self._tf_freq_hz > 0.0
            if not np.any(valid):
                return
            base_values = self._tf_base_values[valid, :]
            floor = np.finfo(np.float64).tiny
            if self._current_tf_scale() == TF_SCALE_LOG:
                if self._tf_base_mode == TF_MODE_PSD:
                    values = 10.0 * np.log10(np.maximum(base_values, floor))
                else:
                    values = 20.0 * np.log10(np.maximum(base_values, floor))
            else:
                values = np.asarray(base_values, dtype=np.float64)
        if values.size == 0:
            return

        if self.tf_color_auto_checkbox.isChecked():
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                return
            if self._tf_color_min_user_override:
                level_min = float(self.tf_color_min_spin.value())
            else:
                level_min = float(self._tf_default_color_min)
            level_max = float(np.nanmax(finite))
            if level_min >= level_max:
                level_min = level_max - 1.0
            self._updating_tf_color_spins = True
            try:
                self.tf_color_min_spin.blockSignals(True)
                self.tf_color_max_spin.blockSignals(True)
                self.tf_color_min_spin.setValue(level_min)
                self.tf_color_max_spin.setValue(level_max)
            finally:
                self.tf_color_min_spin.blockSignals(False)
                self.tf_color_max_spin.blockSignals(False)
                self._updating_tf_color_spins = False
        else:
            level_min = float(self.tf_color_min_spin.value())
            level_max = float(self.tf_color_max_spin.value())
            if level_min >= level_max:
                self.statusBar().showMessage("Invalid t-f color range. Kept previous levels.")
                return

        self.tf_image_item.setLevels((level_min, level_max))
        self.tf_histogram.item.setLevels(level_min, level_max)

    def _clear_short_time_feature_plot(self) -> None:
        self.feature_curve.setData([], [])
        self._apply_feature_y_range()

    def _apply_feature_y_range(self) -> None:
        view_box = self.feature_plot.getViewBox()
        if self._current_feature_mode() == FEATURE_MODE_SVM:
            view_box.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
            view_box.setYRange(-0.1, 1.1, padding=0.0)
            return

        y_min = float(self.feature_y_min_spin.value())
        y_max = float(self.feature_y_max_spin.value())
        if y_min == 0.0 and y_max == 0.0:
            view_box.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            return
        if y_min >= y_max:
            view_box.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            self.statusBar().showMessage("Invalid feature Y range. Switched to auto range.")
            return
        view_box.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
        view_box.setYRange(y_min, y_max, padding=0.0)

    def _rebuild_short_time_feature_plot(self) -> None:
        if self._current_feature_mode() == FEATURE_MODE_SVM:
            self._start_short_time_feature_prediction()
            return
        self._rebuild_short_time_energy_plot()

    def _rebuild_short_time_energy_plot(self) -> None:
        if self._current_waveform is None:
            self._clear_short_time_feature_plot()
            return

        sample_rate = float(self._current_waveform.sample_rate)
        nyquist = sample_rate / 2.0
        band1_low = float(self.feature_num_low_spin.value())
        band1_high = float(self.feature_num_high_spin.value())
        band2_low = float(self.feature_den_low_spin.value())
        band2_high = float(self.feature_den_high_spin.value())
        window_seconds = float(self.feature_window_spin.value())
        hop_ratio = float(self.feature_step_spin.value()) / 100.0
        amplitude_threshold = float(self.feature_amp_threshold_spin.value())

        for low, high, label in (
            (band1_low, band1_high, "Band 1"),
            (band2_low, band2_high, "Band 2"),
        ):
            if low < 0.0 or high <= 0.0 or low >= high:
                self._clear_short_time_feature_plot()
                self.statusBar().showMessage(f"{label} frequency range is invalid.")
                return
            if high >= nyquist:
                self._clear_short_time_feature_plot()
                self.statusBar().showMessage(f"{label} high cutoff must be lower than Nyquist ({nyquist:.1f} Hz).")
                return

        if window_seconds <= 0.0:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage("Short-time window must be greater than 0 s.")
            return
        if hop_ratio <= 0.0 or hop_ratio > 1.0:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage("Step must be in the range (0, 100].")
            return
        if nyquist <= 100.0:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage(
                f"Sample rate is too low for 100 Hz high-pass feature preprocessing (Nyquist {nyquist:.1f} Hz)."
            )
            return

        feature_values = apply_display_filter(
            values=self._current_waveform.phase_data,
            sample_rate=sample_rate,
            enabled=True,
            mode=FilterMode.HIGHPASS,
            low_cut_hz=100.0,
            high_cut_hz=0.0,
        )
        centers, ratio_db = compute_short_time_energy_ratio(
            feature_values,
            sample_rate,
            numerator_low_hz=band1_low,
            numerator_high_hz=band1_high,
            denominator_low_hz=band2_low,
            denominator_high_hz=band2_high,
            window_seconds=window_seconds,
            hop_ratio=hop_ratio,
            amplitude_threshold=amplitude_threshold,
            gate_values=self._current_display_values,
        )
        if centers.size == 0:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage("Short-time feature parameters produced no valid windows.")
            return

        self.feature_curve.setData(centers, ratio_db)
        self._apply_feature_y_range()
        self.statusBar().showMessage(f"Short-time feature updated for {centers.size} windows.")

    def _current_feature_mode(self) -> str:
        return str(self.feature_plot_mode_combo.currentData())

    def _handle_feature_mode_changed(self, _index: int) -> None:
        self._update_feature_plot_style()
        self._rebuild_short_time_feature_plot()

    def _update_feature_plot_style(self) -> None:
        plot_item = self.feature_plot.getPlotItem()
        if self._current_feature_mode() == FEATURE_MODE_SVM:
            plot_item.setLabel("left", "SVM Prediction")
        else:
            plot_item.setLabel("left", "Band Energy Density Ratio (dB)")
        self._apply_feature_y_range()

    def _set_interaction_mode(self, mode: InteractionMode) -> None:
        self.time_plot.set_interaction_mode(mode)
        self.statusBar().showMessage(f"Interaction mode: {mode.value}")

    def _update_interaction_mode(self) -> None:
        if self.zoom_mode_button.isChecked():
            self._set_interaction_mode(InteractionMode.ZOOM)
        else:
            self._set_interaction_mode(InteractionMode.WINDOW_PSD)

    def _zoom_out_time_plot(self) -> None:
        self._push_current_view_to_history()
        self.time_plot.getViewBox().scaleBy((2.0, 1.0))

    def _reset_time_plot(self) -> None:
        if self._current_waveform is None:
            return
        self._push_current_view_to_history()
        self._apply_view_state(
            ((0.0, float(max(1, len(self._current_display_values) - 1))), self._default_y_range())
        )

    def _clear_selection(self) -> None:
        self._set_fixed_psd_enabled(False)
        self.time_plot.clear_selection_region()
        self.psd_curve.setData([], [])
        self.psd_plot.getViewBox().enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
        self._apply_psd_y_range()
        self.window_length_label.setText("Window: 0.000 s")
        self.statusBar().showMessage("Selection cleared.")

    def _toggle_fixed_psd_window(self, checked: bool) -> None:
        self._set_fixed_psd_enabled(checked)
        message = "Fixed PSD window enabled." if checked else "Fixed PSD window disabled."
        self.statusBar().showMessage(message)

    def _set_fixed_psd_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled and self._current_waveform is not None and self._current_display_values.size > 1)
        self._fixed_psd_enabled = enabled
        if self.fixed_psd_button.isChecked() != enabled:
            self.fixed_psd_button.blockSignals(True)
            self.fixed_psd_button.setChecked(enabled)
            self.fixed_psd_button.blockSignals(False)
        if not enabled:
            self._fixed_psd_window_samples = 0
            return

        x_range, _ = self.time_plot.getViewBox().viewRange()
        view_start, view_end = self._normalized_x_range((float(x_range[0]), float(x_range[1])))
        visible_span = max(1.0, view_end - view_start)
        region = self.time_plot._selection_region.getRegion() if self.time_plot._selection_region else None
        if region is not None:
            region_start, region_end = sorted((float(region[0]), float(region[1])))
            region_width = max(1.0, region_end - region_start)
            region_center = (region_start + region_end) * 0.5
            self._fixed_psd_anchor_ratio = min(0.9, max(0.1, (region_center - view_start) / visible_span))
            self._fixed_psd_window_samples = max(self.time_plot._min_window_samples, int(round(region_width)))
        else:
            self._fixed_psd_anchor_ratio = 0.5
            self._fixed_psd_window_samples = max(
                self.time_plot._min_window_samples,
                int(round(max(visible_span * 0.2, self.time_plot._min_window_samples))),
            )
        self._update_fixed_psd_window_from_view()

    def _sync_tf_x_from_time(self, *_args) -> None:
        if self._syncing_time_tf_x or self._current_waveform is None:
            return
        x_range, _ = self.time_plot.getViewBox().viewRange()
        start, end = self._normalized_x_range((float(x_range[0]), float(x_range[1])))
        self._syncing_time_tf_x = True
        try:
            self.tf_plot.setXRange(start, end, padding=0.0)
        finally:
            self._syncing_time_tf_x = False

    def _sync_time_x_from_tf(self, _view_box, x_range) -> None:
        if self._syncing_time_tf_x or self._current_waveform is None:
            return
        start, end = self._normalized_x_range((float(x_range[0]), float(x_range[1])))
        self._syncing_time_tf_x = True
        try:
            _, y_range = self.time_plot.getViewBox().viewRange()
            self._apply_view_state(((start, end), tuple(y_range)))
        finally:
            self._syncing_time_tf_x = False

    def _clamp_time_plot_x_range(self) -> None:
        if self._clamping_time_x_range or self._current_display_values.size <= 1:
            return
        x_range, y_range = self.time_plot.getViewBox().viewRange()
        start, end = self._normalized_x_range((float(x_range[0]), float(x_range[1])))
        if abs(start - float(x_range[0])) < 1e-6 and abs(end - float(x_range[1])) < 1e-6:
            return
        self._clamping_time_x_range = True
        try:
            self._apply_view_state(((start, end), (float(y_range[0]), float(y_range[1]))))
        finally:
            self._clamping_time_x_range = False

    def _handle_time_view_changed(self, *_args) -> None:
        self._clamp_time_plot_x_range()
        if self._fixed_psd_enabled:
            self._update_fixed_psd_window_from_view()

    def _update_fixed_psd_window_from_view(self) -> None:
        if not self._fixed_psd_enabled or self._current_waveform is None or self._current_display_values.size <= 1:
            return

        x_range, _ = self.time_plot.getViewBox().viewRange()
        view_start, view_end = self._normalized_x_range((float(x_range[0]), float(x_range[1])))
        visible_span = max(1.0, view_end - view_start)
        window_samples = max(1, self._fixed_psd_window_samples)
        window_samples = max(window_samples, self.time_plot._min_window_samples)
        window_samples = min(
            window_samples,
            max(1, int(round(visible_span))),
            self._current_display_values.size - 1,
        )
        center = view_start + visible_span * self._fixed_psd_anchor_ratio
        half_width = window_samples / 2.0
        start = int(round(center - half_width))
        max_start = max(0, self._current_display_values.size - 1 - window_samples)
        start = min(max(0, start), max_start)
        end = min(self._current_display_values.size - 1, start + window_samples)
        if end <= start:
            end = min(self._current_display_values.size - 1, start + 1)
        self.time_plot.set_selection_region(start, end)
        self._fixed_psd_window_samples = max(1, end - start)
        self._update_psd_from_selection(start, end)

    def _update_psd_from_selection(self, start_index: int, end_index: int) -> None:
        if self._current_waveform is None:
            return

        raw_values = self._current_waveform.phase_data[start_index:end_index]
        freqs, psd_db = compute_window_psd(raw_values, self._current_waveform.sample_rate)
        valid = freqs > 0.0
        freqs = freqs[valid]
        psd_db = psd_db[valid]
        if freqs.size == 0:
            self.psd_curve.setData([], [])
            self.statusBar().showMessage("Selection window is too short for PSD.")
            return

        self.psd_curve.setData(freqs, psd_db)
        nyquist = self._current_waveform.sample_rate / 2.0
        lower_hz = max(1.0, min(1000.0, nyquist))
        self.psd_plot.setXRange(np.log10(lower_hz), np.log10(nyquist), padding=0.0)
        self._apply_psd_y_range()
        window_seconds = max(0.0, (end_index - start_index) / self._current_waveform.sample_rate)
        self.window_length_label.setText(f"Window: {window_seconds:.6f} s")
        self.statusBar().showMessage(
            f"PSD updated for samples {start_index} to {end_index}."
        )

    def _record_view_history(self, _, view_range) -> None:
        if self._suspend_history:
            return
        x_range = tuple(float(value) for value in view_range[0])
        y_range = tuple(float(value) for value in view_range[1])
        new_state = (x_range, y_range)
        if self._last_view_state is None:
            self._last_view_state = new_state
            return
        if self._states_close(new_state, self._last_view_state):
            return
        self._view_history.append(self._last_view_state)
        if len(self._view_history) > 30:
            self._view_history.pop(0)
        self._last_view_state = new_state
        self.back_view_button.setEnabled(bool(self._view_history))

    def _push_current_view_to_history(self) -> None:
        state = self._current_view_state()
        if state is None:
            return
        if self._last_view_state is None:
            self._last_view_state = state
        if self._view_history and self._states_close(self._view_history[-1], state):
            return
        if self._last_view_state and self._states_close(self._last_view_state, state):
            self._view_history.append(state)
        else:
            self._view_history.append(state)
            self._last_view_state = state
        if len(self._view_history) > 30:
            self._view_history.pop(0)
        self.back_view_button.setEnabled(bool(self._view_history))

    def _restore_previous_view(self) -> None:
        if not self._view_history:
            return
        state = self._view_history.pop()
        self._apply_view_state(state)
        self.back_view_button.setEnabled(bool(self._view_history))
        self.statusBar().showMessage("Returned to previous view.")

    def _clear_view_history(self) -> None:
        self._view_history.clear()
        self._last_view_state = None
        self.back_view_button.setEnabled(False)

    def _current_view_state(self) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
        if self._current_waveform is None:
            return None
        x_range, y_range = self.time_plot.getViewBox().viewRange()
        return (
            (float(x_range[0]), float(x_range[1])),
            (float(y_range[0]), float(y_range[1])),
        )

    def _default_y_range(self) -> tuple[float, float]:
        y_min = float(self.y_min_spin.value())
        y_max = float(self.y_max_spin.value())
        if y_min != 0.0 or y_max != 0.0:
            if y_min < y_max:
                return (y_min, y_max)

        if self._current_display_values.size == 0:
            return (-1.0, 1.0)
        data_min = float(np.min(self._current_display_values))
        data_max = float(np.max(self._current_display_values))
        if data_min == data_max:
            pad = max(1.0, abs(data_min) * 0.1)
            return (data_min - pad, data_max + pad)
        pad = max((data_max - data_min) * 0.02, 1e-9)
        return (data_min - pad, data_max + pad)

    def _apply_view_state(
        self, state: tuple[tuple[float, float], tuple[float, float]]
    ) -> None:
        self._suspend_history = True
        try:
            x_range, y_range = state
            x_range = self._normalized_x_range(x_range)
            self.time_plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
            self.time_plot.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
            self.time_plot.setXRange(x_range[0], x_range[1], padding=0.0)
            self.time_plot.getViewBox().setYRange(y_range[0], y_range[1], padding=0.0)
        finally:
            self._suspend_history = False
        self._last_view_state = state
        self._refresh_length_labels()

    def _update_visible_length_label(self, *_args) -> None:
        self._refresh_length_labels()

    def _apply_visible_window_duration(self) -> None:
        if self._current_waveform is None or self._current_display_values.size <= 1:
            return

        requested_seconds = float(self.visible_window_spin.value())
        if requested_seconds <= 0.0:
            self.statusBar().showMessage('Visible window duration must be greater than 0 s.')
            return

        sample_rate = max(self._current_waveform.sample_rate, 1.0)
        requested_samples = max(1.0, requested_seconds * sample_rate)
        x_range, y_range = self.time_plot.getViewBox().viewRange()
        center = (float(x_range[0]) + float(x_range[1])) * 0.5
        half_width = requested_samples * 0.5
        self._apply_view_state(((center - half_width, center + half_width), tuple(y_range)))

    def _refresh_length_labels(self) -> None:
        if self._current_waveform is None:
            self.visible_length_label.setText("Visible: 0.000 s")
            self._update_time_scrollbar()
            return
        x_range, _ = self.time_plot.getViewBox().viewRange()
        start, end = self._normalized_x_range((float(x_range[0]), float(x_range[1])))
        sample_rate = max(self._current_waveform.sample_rate, 1.0)
        visible_seconds = max(0.0, (end - start) / sample_rate)
        self.visible_length_label.setText(f"Visible: {visible_seconds:.3f} s")
        self.visible_window_spin.blockSignals(True)
        self.visible_window_spin.setValue(max(round(visible_seconds, 3), self.visible_window_spin.minimum()))
        self.visible_window_spin.blockSignals(False)
        self._refresh_default_audio_path()
        self._update_time_scrollbar()

    def _apply_psd_y_range(self) -> None:
        view_box = self.psd_plot.getViewBox()
        y_min = int(self.psd_y_min_spin.value())
        y_max = int(self.psd_y_max_spin.value())
        if y_min >= y_max:
            self.statusBar().showMessage("Invalid PSD Y range. Kept previous range.")
            return
        view_box.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
        view_box.setRange(yRange=(float(y_min), float(y_max)), padding=0.0, disableAutoRange=True)

    def _normalized_x_range(self, x_range: tuple[float, float]) -> tuple[float, float]:
        if self._current_display_values.size <= 1:
            return (0.0, 1.0)

        data_max = float(self._current_display_values.size - 1)
        start, end = sorted((float(x_range[0]), float(x_range[1])))
        width = max(1.0, end - start)
        width = min(width, max(1.0, data_max))
        start = min(max(0.0, start), max(0.0, data_max - width))
        end = min(data_max, start + width)
        start = max(0.0, end - width)
        return (start, end)

    def _update_time_scrollbar(self) -> None:
        self._syncing_scrollbar = True
        try:
            if self._current_display_values.size <= 1:
                self.time_scrollbar.setEnabled(False)
                self.time_scrollbar.setRange(0, 0)
                self.time_scrollbar.setPageStep(1)
                self.time_scrollbar.setValue(0)
                return

            x_range, _ = self.time_plot.getViewBox().viewRange()
            start, end = self._normalized_x_range((float(x_range[0]), float(x_range[1])))
            total_span = max(1, self._current_display_values.size - 1)
            visible_span = max(1, int(round(end - start)))
            max_start = max(0, total_span - visible_span)
            scrollbar_value = int(round(min(max(start, 0.0), float(max_start))))

            self.time_scrollbar.setEnabled(max_start > 0)
            self.time_scrollbar.setRange(0, max_start)
            self.time_scrollbar.setPageStep(visible_span)
            self.time_scrollbar.setSingleStep(max(1, visible_span // 10))
            self.time_scrollbar.setValue(scrollbar_value)
        finally:
            self._syncing_scrollbar = False

    def _handle_time_scrollbar_change(self, value: int) -> None:
        if self._syncing_scrollbar or self._current_waveform is None:
            return
        x_range, y_range = self.time_plot.getViewBox().viewRange()
        visible_span = max(1.0, float(x_range[1]) - float(x_range[0]))
        self._apply_view_state(((float(value), float(value) + visible_span), tuple(y_range)))

    def _scroll_time_plot_by_step(self, direction: int) -> None:
        if self._current_waveform is None or self._current_display_values.size <= 1:
            return
        step = max(1, self.time_scrollbar.singleStep())
        new_value = self.time_scrollbar.value() + direction * step
        new_value = min(max(new_value, self.time_scrollbar.minimum()), self.time_scrollbar.maximum())
        if new_value != self.time_scrollbar.value():
            self.time_scrollbar.setValue(new_value)

    @staticmethod
    def _states_close(
        left: tuple[tuple[float, float], tuple[float, float]],
        right: tuple[tuple[float, float], tuple[float, float]],
    ) -> bool:
        return all(
            abs(a - b) < 1e-6
            for pair_left, pair_right in zip(left, right)
            for a, b in zip(pair_left, pair_right)
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._audio_player.stop()
        self._clear_audio_temp_path()
        super().closeEvent(event)
