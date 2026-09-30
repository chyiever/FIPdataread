"""Plot 2 short-time features and the sliding-window SVM prediction.

Part of the ``MainWindow`` mixin set; see ``main_window`` for the full table of
responsibilities.
"""

from PyQt5 import QtCore
from constants import FEATURE_MODE_BAND_ENERGY
from constants import FEATURE_MODE_CHANNEL_1
from constants import FEATURE_MODE_CHANNEL_2
from constants import FEATURE_MODE_ENERGY
from constants import FEATURE_MODE_ENERGY_ENERGY
from constants import FEATURE_MODE_MAX_NUM
from constants import FEATURE_MODE_NONE
from constants import FEATURE_MODE_PSD_SUM
from constants import FEATURE_MODE_SVM
from models import FilterMode
from processing import apply_display_filter
from processing import compute_short_time_band_energy
from processing import compute_short_time_energy_ratio
from processing import compute_short_time_energy_sum
from processing import compute_short_time_max_num
from processing import compute_short_time_psd_sum
from widgets import SVMPredictionWorker
import numpy as np
import pyqtgraph as pg


class ShortTimeFeaturePanelMixin:
    """Plot 2 short-time features and the sliding-window SVM prediction.

    Covers the five short-time features, the raw waveform view, and the background
    SVM prediction thread that produces the ``SVM Prediction`` mode.
    """

    def _start_short_time_feature_prediction(self) -> None:
        """Start the background SVM prediction for Plot 2.

        The request is stamped with an incrementing task id, which the finished
        and failed handlers compare against the current id so that a superseded
        request's result is discarded instead of being drawn.
        """

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
        """Tear down the SVM prediction thread and worker.
        """

        self._prediction_thread = None
        self._prediction_worker = None


    @QtCore.pyqtSlot(int, object, object)
    def _handle_prediction_finished(self, task_id: int, centers: np.ndarray, predictions: np.ndarray) -> None:
        """Adopt the SVM predictions and draw them into Plot 2.
        """

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
        """Report an SVM prediction failure in the status bar.
        """

        if task_id != self._prediction_task_id or self._current_feature_mode() != FEATURE_MODE_SVM:
            return
        self._clear_short_time_feature_plot()
        self.statusBar().showMessage(f'Failed to compute SVM prediction: {message}')


    def _clear_short_time_feature_plot(self) -> None:
        """Remove the current Plot 2 curve.
        """

        self.feature_curve.setData([], [])
        self._apply_feature_y_range()


    def _apply_feature_y_range(self) -> None:
        """Apply the shared feature Y range (``0 / 0`` means auto).
        """

        view_box = self.feature_plot.getViewBox()
        if self._current_feature_mode() == FEATURE_MODE_NONE:
            view_box.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            return
        if self._current_feature_mode() == FEATURE_MODE_SVM:
            view_box.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
            view_box.setYRange(-0.1, 1.1, padding=0.0)
            return
        if self._current_feature_mode() in {FEATURE_MODE_CHANNEL_1, FEATURE_MODE_CHANNEL_2}:
            y_min = float(self.y_min_spin.value())
            y_max = float(self.y_max_spin.value())
            if y_min == 0.0 and y_max == 0.0:
                view_box.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
                return
            if y_min >= y_max:
                view_box.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
                self.statusBar().showMessage("Invalid phase Y range. Switched Plot 2 to auto range.")
                return
            view_box.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
            view_box.setYRange(y_min, y_max, padding=0.0)
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
        """Dispatch to the builder for the selected Plot 2 mode.
        """

        if self._current_feature_mode() == FEATURE_MODE_NONE:
            self._clear_short_time_feature_plot()
            return
        if self._current_feature_mode() in {FEATURE_MODE_CHANNEL_1, FEATURE_MODE_CHANNEL_2}:
            channel_index = 0 if self._current_feature_mode() == FEATURE_MODE_CHANNEL_1 else 1
            self._rebuild_channel_waveform_plot(channel_index)
            return
        if self._current_feature_mode() == FEATURE_MODE_SVM:
            self._start_short_time_feature_prediction()
            return
        if self._current_feature_mode() == FEATURE_MODE_ENERGY:
            self._rebuild_short_time_energy_ratio_plot()
            return
        if self._current_feature_mode() == FEATURE_MODE_BAND_ENERGY:
            self._rebuild_short_time_band_energy_plot()
            return
        if self._current_feature_mode() == FEATURE_MODE_ENERGY_ENERGY:
            self._rebuild_short_time_energy_energy_plot()
            return
        if self._current_feature_mode() == FEATURE_MODE_PSD_SUM:
            self._rebuild_short_time_psd_sum_plot()
            return
        self._rebuild_short_time_max_num_plot()


    def _rebuild_channel_waveform_plot(self, channel_index: int) -> None:
        """Draw a raw CH1/CH2 waveform into Plot 2.

        Uses the same display-filter preprocessing as the top time plot, so the two
        time-domain plots stay visually consistent.
        """

        if self._current_waveform is None or self._current_waveform.channel_count <= channel_index:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage(f"CH{channel_index + 1} is not available for the current file.")
            return

        values = self._build_channel_display_values(self._current_waveform, channel_index)
        if values is None:
            self._clear_short_time_feature_plot()
            return

        self.feature_curve.setData(values)
        self._apply_feature_y_range()
        self.statusBar().showMessage(
            f"{self._current_waveform.channel_label(channel_index)} waveform updated in Plot 2."
        )


    def _rebuild_short_time_energy_ratio_plot(self) -> None:
        """Draw the ST Energy Ratio feature curve (dB).
        """

        if self._current_waveform is None:
            self._clear_short_time_feature_plot()
            return

        sample_rate = float(self._current_waveform.sample_rate)
        nyquist = sample_rate / 2.0
        band1_low = float(self.feature_num_low_spin.value())
        band1_high = float(self.feature_num_high_spin.value())
        band2_low = float(self.feature_den_low_spin.value())
        band2_high = float(self.feature_den_high_spin.value())
        window_seconds = float(self.feature_window_spin.value()) / 1000.0
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
            self.statusBar().showMessage("Short-time window must be greater than 0 ms.")
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


    def _rebuild_short_time_band_energy_plot(self) -> None:
        """Draw the ST Energy feature curve (linear).
        """

        if self._current_waveform is None:
            self._clear_short_time_feature_plot()
            return

        sample_rate = float(self._current_waveform.sample_rate)
        nyquist = sample_rate / 2.0
        band_low = float(self.feature_energy_band_low_spin.value())
        band_high = float(self.feature_energy_band_high_spin.value())
        window_seconds = float(self.feature_window_spin.value()) / 1000.0
        hop_ratio = float(self.feature_step_spin.value()) / 100.0
        amplitude_threshold = float(self.feature_amp_threshold_spin.value())

        if band_low < 0.0 or band_high <= 0.0 or band_low >= band_high:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage("Band frequency range is invalid.")
            return
        if band_high >= nyquist:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage(f"Band high cutoff must be lower than Nyquist ({nyquist:.1f} Hz).")
            return
        if window_seconds <= 0.0:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage("Short-time window must be greater than 0 ms.")
            return
        if hop_ratio <= 0.0 or hop_ratio > 1.0:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage("Step must be in the range (0, 100].")
            return

        centers, energies = compute_short_time_band_energy(
            values=self._current_waveform.phase_data,
            sample_rate=sample_rate,
            band_low_hz=band_low,
            band_high_hz=band_high,
            window_seconds=window_seconds,
            hop_ratio=hop_ratio,
            amplitude_threshold=amplitude_threshold,
            gate_values=self._current_display_values,
        )
        if centers.size == 0:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage("Short-time feature parameters produced no valid windows.")
            return

        self.feature_curve.setData(centers, energies)
        self._apply_feature_y_range()
        self.statusBar().showMessage(f"Short-time feature updated for {centers.size} windows.")


    def _rebuild_short_time_energy_energy_plot(self) -> None:
        """Draw the ST Energy Energy feature curve (linear).
        """

        if self._current_waveform is None:
            self._clear_short_time_feature_plot()
            return

        sample_rate = float(self._current_waveform.sample_rate)
        nyquist = sample_rate / 2.0
        band_low = float(self.feature_energy2_band_low_spin.value())
        band_high = float(self.feature_energy2_band_high_spin.value())
        stage1_window = float(self.feature_energy2_window_spin.value()) / 1000.0
        stage1_step = float(self.feature_energy2_step_spin.value()) / 100.0
        amplitude_threshold = float(self.feature_energy2_amp_threshold_spin.value())
        stage2_window = float(self.feature_energy2_sum_window_spin.value()) / 1000.0
        stage2_step = float(self.feature_energy2_sum_step_spin.value()) / 100.0

        if band_low < 0.0 or band_high <= 0.0 or band_low >= band_high:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage("Band frequency range is invalid.")
            return
        if band_high >= nyquist:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage(f"Band high cutoff must be lower than Nyquist ({nyquist:.1f} Hz).")
            return
        if stage1_window <= 0.0 or stage2_window <= 0.0:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage("Stage windows must be greater than 0 ms.")
            return
        if stage1_step <= 0.0 or stage1_step > 1.0 or stage2_step <= 0.0 or stage2_step > 1.0:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage("Stage steps must be in the range (0, 100].")
            return

        centers1, energies1 = compute_short_time_band_energy(
            values=self._current_waveform.phase_data,
            sample_rate=sample_rate,
            band_low_hz=band_low,
            band_high_hz=band_high,
            window_seconds=stage1_window,
            hop_ratio=stage1_step,
            amplitude_threshold=amplitude_threshold,
            gate_values=self._current_display_values,
        )
        if centers1.size == 0:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage("Short-time feature parameters produced no valid windows.")
            return

        centers2, sums = compute_short_time_energy_sum(
            centers1,
            energies1,
            window_seconds=stage2_window,
            hop_ratio=stage2_step,
            sample_rate=sample_rate,
        )
        if centers2.size == 0:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage("Stage 2 parameters produced no valid windows.")
            return

        self.feature_curve.setData(centers2, sums)
        self._apply_feature_y_range()
        self.statusBar().showMessage(f"Short-time feature updated for {centers2.size} windows.")


    def _rebuild_short_time_psd_sum_plot(self) -> None:
        """Draw the ST PSD sum feature curve (rad^2/Hz).
        """

        if self._current_waveform is None:
            self._clear_short_time_feature_plot()
            return

        sample_rate = float(self._current_waveform.sample_rate)
        nyquist = sample_rate / 2.0
        band_low = float(self.feature_psd_sum_band_low_spin.value())
        band_high = float(self.feature_psd_sum_band_high_spin.value())
        psd_window = float(self.feature_psd_sum_window_spin.value())
        hop_seconds = float(self.feature_psd_sum_step_spin.value())
        background_seconds = float(self.feature_psd_sum_background_spin.value())

        if band_low < 0.0 or band_high <= 0.0 or band_low >= band_high:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage("Band frequency range is invalid.")
            return
        if band_high > nyquist:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage(f"Band high cutoff must not exceed Nyquist ({nyquist:.1f} Hz).")
            return
        if psd_window <= 0.0 or hop_seconds <= 0.0 or background_seconds <= 0.0:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage("PSD window, step, and background window must be greater than 0 s.")
            return

        centers, sums = compute_short_time_psd_sum(
            values=self._current_waveform.phase_data,
            sample_rate=sample_rate,
            psd_window_seconds=psd_window,
            hop_seconds=hop_seconds,
            background_seconds=background_seconds,
            band_low_hz=band_low,
            band_high_hz=band_high,
        )
        if centers.size == 0:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage("Short-time feature parameters produced no valid windows.")
            return

        self.feature_curve.setData(centers, sums)
        self._apply_feature_y_range()
        self.statusBar().showMessage(f"Short-time feature updated for {centers.size} windows.")


    def _rebuild_short_time_max_num_plot(self) -> None:
        """Draw the ST-energy-max-num feature curve.
        """

        if self._current_waveform is None:
            self._clear_short_time_feature_plot()
            return

        sample_rate = float(self._current_waveform.sample_rate)
        nyquist = sample_rate / 2.0
        band_low = float(self.feature_maxnum_band_low_spin.value())
        band_high = float(self.feature_maxnum_band_high_spin.value())
        stage1_window = float(self.feature_maxnum_window_spin.value()) / 1000.0
        stage1_step = float(self.feature_maxnum_step_spin.value()) / 100.0
        amplitude_threshold = float(self.feature_maxnum_amp_threshold_spin.value())
        stage2_window = float(self.feature_maxnum_sum_window_spin.value()) / 1000.0
        stage2_step = float(self.feature_maxnum_sum_step_spin.value())
        sub_window = float(self.feature_maxnum_sub_window_spin.value()) / 1000.0
        max_threshold = float(self.feature_maxnum_threshold_spin.value()) * 1e-6

        if band_low < 0.0 or band_high <= 0.0 or band_low >= band_high:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage("Band frequency range is invalid.")
            return
        if band_high >= nyquist:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage(f"Band high cutoff must be lower than Nyquist ({nyquist:.1f} Hz).")
            return
        if stage1_window <= 0.0 or stage2_window <= 0.0 or sub_window <= 0.0:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage("Stage and sub windows must be greater than 0 ms.")
            return
        if stage1_step <= 0.0 or stage1_step > 1.0 or stage2_step <= 0.0:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage("Stage 1 step must be in (0, 100] and Stage 2 step must be greater than 0.")
            return

        centers1, energies1 = compute_short_time_band_energy(
            values=self._current_waveform.phase_data,
            sample_rate=sample_rate,
            band_low_hz=band_low,
            band_high_hz=band_high,
            window_seconds=stage1_window,
            hop_ratio=stage1_step,
            amplitude_threshold=amplitude_threshold,
            gate_values=self._current_display_values,
        )
        if centers1.size == 0:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage("Short-time feature parameters produced no valid windows.")
            return

        centers2, counts = compute_short_time_max_num(
            centers1,
            energies1,
            window_seconds=stage2_window,
            hop_seconds=stage2_step,
            sub_window_seconds=sub_window,
            max_threshold=max_threshold,
            sample_rate=sample_rate,
        )
        if centers2.size == 0:
            self._clear_short_time_feature_plot()
            self.statusBar().showMessage("Stage 2 parameters produced no valid windows.")
            return

        self.feature_curve.setData(centers2, counts)
        self._apply_feature_y_range()
        self.statusBar().showMessage(f"Short-time feature updated for {centers2.size} windows.")


    def _current_feature_mode(self) -> str:
        """Return the identifier of the selected Plot 2 mode.
        """

        return str(self.feature_plot_mode_combo.currentData())


    def _handle_feature_mode_changed(self, _index: int) -> None:
        """Switch the feature parameter page and redraw Plot 2.
        """

        self._update_feature_plot_style()
        self._update_curve_splitter_for_feature_mode()
        self._update_feature_params_page()
        self._rebuild_short_time_feature_plot()


    def _update_feature_params_page(self) -> None:
        """Show the parameter page matching the Plot 2 mode.

        Hides the shared window/step/gate controls on pages that do not use them.
        """

        mode = self._current_feature_mode()
        if mode == FEATURE_MODE_ENERGY:
            index = 0
            show_window_gate = True
        elif mode == FEATURE_MODE_BAND_ENERGY:
            index = 1
            show_window_gate = True
        elif mode == FEATURE_MODE_ENERGY_ENERGY:
            index = 2
            show_window_gate = False
        elif mode == FEATURE_MODE_PSD_SUM:
            index = 3
            show_window_gate = False
        elif mode == FEATURE_MODE_MAX_NUM:
            index = 4
            show_window_gate = False
        else:
            index = 0
            show_window_gate = True
        self.feature_params_stack.setCurrentIndex(index)
        self.feature_window_gate_widget.setVisible(show_window_gate)


    def _update_feature_plot_style(self) -> None:
        """Set the Plot 2 Y-axis label for the current mode.
        """

        plot_item = self.feature_plot.getPlotItem()
        if self._current_feature_mode() == FEATURE_MODE_NONE:
            plot_item.setLabel("left", "Plot 2")
        elif self._current_feature_mode() in {FEATURE_MODE_CHANNEL_1, FEATURE_MODE_CHANNEL_2}:
            plot_item.setLabel("left", "Phase (rad)")
        elif self._current_feature_mode() == FEATURE_MODE_SVM:
            plot_item.setLabel("left", "SVM Prediction")
        else:
            plot_item.setLabel("left", "Feature value")
        self._apply_feature_y_range()

