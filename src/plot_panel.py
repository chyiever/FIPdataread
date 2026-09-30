"""Time-domain plotting, Welch PSD, interaction modes and view history.

Part of the ``MainWindow`` mixin set; see ``main_window`` for the full table of
responsibilities.
"""

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from constants import FEATURE_MODE_CHANNEL_2
from constants import FEATURE_MODE_NONE
from constants import PSD_SOURCE_BOTH
from constants import PSD_SOURCE_CHANNEL_1
from constants import PSD_SOURCE_CHANNEL_2
from constants import TF_SOURCE_CHANNEL_2
from datetime import timedelta
from models import InteractionMode
from models import LoadedWaveform
from plotting import AbsoluteTimeAxis
from processing import apply_display_filter
from processing import compute_window_psd
from processing import validate_filter
from typing import Optional
import numpy as np
import pyqtgraph as pg


class PlotPanelMixin:
    """Time-domain plotting, Welch PSD, interaction modes and view history.

    Owns the top time plot, Plot 2 and the PSD plot, including the axis
    synchronisation between them, the sticky PSD window and the undo/redo stack of
    view states.
    """

    def _set_combo_item_enabled(self, combo: QtWidgets.QComboBox, item_data: str, enabled: bool) -> None:
        """Enable/disable one combo item by its stored ``item_data``.

        ``QComboBox`` has no API for disabling a single item, so the item is
        looked up in the combo's model and its enabled flag is set instead. The
        first matching item is used.
        """

        model = combo.model()
        for index in range(combo.count()):
            if combo.itemData(index) != item_data:
                continue
            item = model.item(index) if hasattr(model, "item") else None
            if item is not None:
                item.setEnabled(enabled)
            return


    def _has_channel_2(self) -> bool:
        """True when the loaded waveform carries at least two channels.
        """

        return self._current_waveform is not None and self._current_waveform.channel_count >= 2


    def _update_channel_option_controls(self) -> None:
        """Enable or disable the CH2 options for the loaded file.
        """

        has_channel_2 = self._has_channel_2()
        self._set_combo_item_enabled(self.feature_plot_mode_combo, FEATURE_MODE_CHANNEL_2, has_channel_2)
        self._set_combo_item_enabled(self.psd_source_combo, PSD_SOURCE_CHANNEL_2, has_channel_2)
        self._set_combo_item_enabled(self.psd_source_combo, PSD_SOURCE_BOTH, has_channel_2)
        self._set_combo_item_enabled(self.tf_source_combo, TF_SOURCE_CHANNEL_2, has_channel_2)

        if has_channel_2 and self._current_feature_mode() == FEATURE_MODE_NONE:
            self.feature_plot_mode_combo.setCurrentIndex(2)
        if not has_channel_2 and self._current_feature_mode() != FEATURE_MODE_NONE:
            self.feature_plot_mode_combo.setCurrentIndex(0)
        if not has_channel_2 and self._current_psd_source() in {PSD_SOURCE_CHANNEL_2, PSD_SOURCE_BOTH}:
            self.psd_source_combo.setCurrentIndex(0)
        if not has_channel_2 and self._current_tf_source() == TF_SOURCE_CHANNEL_2:
            self.tf_source_combo.setCurrentIndex(0)
        self._update_curve_splitter_for_feature_mode()


    def _current_psd_source(self) -> str:
        """Return the selected PSD channel source identifier.
        """

        return str(self.psd_source_combo.currentData() or PSD_SOURCE_CHANNEL_1)


    def _selected_psd_channel_indices(self) -> list[int]:
        """Map the PSD source identifier to channel indices.

        ``both`` yields ``(0, 1)``; the result is truncated to the channels actually
        present so single-channel files can still request ``CH1+CH2``.
        """

        if self._current_waveform is None:
            return []
        source = self._current_psd_source()
        if source == PSD_SOURCE_CHANNEL_2 and self._current_waveform.channel_count >= 2:
            return [1]
        if source == PSD_SOURCE_BOTH and self._current_waveform.channel_count >= 2:
            return [0, 1]
        return [0]


    def _clear_psd_plot(self) -> None:
        """Remove any existing PSD curve from the PSD plot.
        """

        self.psd_curve.setData([], [])
        self.psd_curve_channel_2.setData([], [])


    def _handle_psd_source_changed(self, _index: int) -> None:
        """Recompute the PSD after the PSD source combo changes.
        """

        if self._current_waveform is None:
            self._clear_psd_plot()
            return
        region = self.time_plot._selection_region.getRegion() if self.time_plot._selection_region else None
        if region is None:
            self._clear_psd_plot()
            return
        start_index, end_index = sorted((int(round(float(region[0]))), int(round(float(region[1])))))
        self._update_psd_from_selection(start_index, end_index)


    def _format_cursor_time(self, sample_index: float) -> Optional[str]:
        """Format a sample index as an absolute ``HH:MM:SS.mmm`` time.
        """

        if self._current_waveform is None:
            return None
        bounded_sample = float(sample_index)
        if self._current_display_values.size > 1:
            bounded_sample = min(max(bounded_sample, 0.0), float(self._current_display_values.size - 1))
        sample_rate = max(float(self._current_waveform.sample_rate), 1.0)
        timestamp = self._current_waveform.start_time + timedelta(seconds=bounded_sample / sample_rate)
        return timestamp.strftime("%H:%M:%S.%f")[:-3]


    def _handle_time_plot_mouse_moved(self, scene_pos: QtCore.QPointF) -> None:
        """Update the cursor readout under the time-domain plot.
        """

        if self._current_waveform is None:
            return
        view_box = self.time_plot.getViewBox()
        if not view_box.sceneBoundingRect().contains(scene_pos):
            return
        point = view_box.mapSceneToView(scene_pos)
        x = float(point.x())
        y = float(point.y())
        if self._current_display_values.size > 1:
            x = min(max(x, 0.0), float(self._current_display_values.size - 1))
        time_text = self._format_cursor_time(x)
        if time_text is None:
            return
        self.statusBar().showMessage(f"Cursor(Time): t={time_text}, amplitude={y:.6g}")


    def _build_channel_display_values(
        self, waveform: Optional[LoadedWaveform], channel_index: int, *, strict_validation: bool = False
    ) -> Optional[np.ndarray]:
        """Return the display-filtered data for one channel.

        Validates the filter against the file's sample rate first; when the filter is
        invalid the message is shown in the status bar and the unfiltered values are
        returned, unless ``strict_validation`` asks for ``None`` instead.
        """

        if waveform is None:
            return None

        try:
            values = waveform.channel_data(channel_index)
        except IndexError:
            self.statusBar().showMessage(f"Channel {channel_index + 1} is not available.")
            return None

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


    def _build_display_values(
        self, waveform: Optional[LoadedWaveform], *, strict_validation: bool = False
    ) -> Optional[np.ndarray]:
        """Return the display values for the top time-domain plot.
        """

        return self._build_channel_display_values(waveform, 0, strict_validation=strict_validation)


    def _get_display_values(self) -> Optional[np.ndarray]:
        """Return the display values for channel 1 of the current waveform.

        Recomputed on each call rather than cached, so it always reflects the
        current filter settings.
        """

        return self._build_display_values(self._current_waveform)


    def _rebuild_time_plot(self) -> None:
        """Redraw the top time-domain plot from the current display values.

        Rebuilds the whole plot context for a newly loaded file: pushes the
        absolute-time context into every axis, resets the selection regions, the
        PSD, the t-f view and the view history, and then repaints Plot 2, the t-f
        time plot and the t-f map. Clipping and peak downsampling are configured
        once per curve in ``_build_ui``, so this only feeds the new data in.
        """

        if self._current_waveform is None:
            return

        values = self._get_display_values()
        if values is None:
            return

        self._current_display_values = values
        self._refresh_time_plot_channel()
        for plot_widget in (self.time_plot, self.feature_plot, self.tf_time_plot, self.tf_plot):
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
        self.feature_plot.set_data_context(
            data_length=len(values),
            sample_rate=self._current_waveform.sample_rate,
            min_window_seconds=0.001,
        )
        self._set_fixed_psd_enabled(False)
        self.time_plot.clear_selection_region()
        self.feature_plot.clear_selection_region()
        self._clear_short_time_feature_plot()
        self._clear_psd_plot()
        self._clear_time_frequency_plot()
        self.window_length_label.setText("Window: 0.000 s")
        self._clear_view_history()
        self._apply_view_state(
            ((0.0, float(max(1, len(self._current_display_values) - 1))), self._default_y_range())
        )
        if self._arrival_sample_index is not None:
            self._set_arrival_marker(self._arrival_sample_index, announce=False)
        self._refresh_default_audio_path()
        self._rebuild_short_time_feature_plot()
        self._rebuild_tf_time_plot()
        self._rebuild_time_frequency_plot()


    def _refresh_time_plot_channel(self) -> None:
        """Redraw the top time plot after the channel source changed.
        """

        if self._current_waveform is None:
            self.time_curve.setData([])
            return

        values = self._build_tf_display_values()
        if values is None or values.size == 0:
            values = self._current_display_values
        self.time_curve.setData(values)


    def _apply_y_range(self) -> None:
        """Apply the phase Y range, using auto-scaling when both values are 0.
        """

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
        self._apply_psd_x_range()
        self._apply_psd_y_range()
        self._apply_feature_y_range()
        self._apply_time_frequency_y_range()
        self._apply_time_frequency_color_levels()


    def _set_interaction_mode(self, mode: InteractionMode) -> None:
        """Set the interaction mode on the time plots and update the toolbar.
        """

        self.time_plot.set_interaction_mode(mode)
        self.feature_plot.set_interaction_mode(mode)
        self.statusBar().showMessage(f"Interaction mode: {mode.value}")


    def _update_interaction_mode(self) -> None:
        """Reflect the current interaction mode in the mode buttons.
        """

        if self.zoom_mode_button.isChecked():
            self._set_interaction_mode(InteractionMode.ZOOM)
        else:
            self._set_interaction_mode(InteractionMode.WINDOW_PSD)


    def _zoom_out_time_plot(self) -> None:
        """Halve the visible span around the current centre.
        """

        self._push_current_view_to_history()
        x_range, y_range = self.time_plot.getViewBox().viewRange()
        center = (float(x_range[0]) + float(x_range[1])) * 0.5
        half_width = max(float(x_range[1]) - float(x_range[0]), 1.0)
        self._apply_view_state(((center - half_width, center + half_width), tuple(y_range)))


    def _reset_time_plot(self) -> None:
        """Restore the full time span and the default Y range.
        """

        if self._current_waveform is None:
            return
        self._push_current_view_to_history()
        self._apply_view_state(
            ((0.0, float(max(1, len(self._current_display_values) - 1))), self._default_y_range())
        )


    def _clear_selection(self) -> None:
        """Clear the active window selection on all time plots.
        """

        self._set_fixed_psd_enabled(False)
        self.time_plot.clear_selection_region()
        self.feature_plot.clear_selection_region()
        self._clear_psd_plot()
        self.psd_plot.getViewBox().enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
        self._apply_psd_y_range()
        self.window_length_label.setText("Window: 0.000 s")
        self.statusBar().showMessage("Selection cleared.")


    def _toggle_fixed_psd_window(self, checked: bool) -> None:
        """Enable/disable the sticky PSD window feature.
        """

        self._set_fixed_psd_enabled(checked)
        message = "Fixed PSD window enabled." if checked else "Fixed PSD window disabled."
        self.statusBar().showMessage(message)


    def _set_fixed_psd_enabled(self, enabled: bool) -> None:
        """Turn the fixed PSD window on or off.

        When enabled the selection no longer follows the view; the window is
        re-anchored to a stored fraction of the *visible* span, so it keeps
        pointing at the same part of the signal while the user pans.
        """

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
        """Propagate the time plot's X range to the t-f plot.
        """

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
        """Propagate the t-f plot's X range to the time plot.
        """

        if self._syncing_time_tf_x or self._current_waveform is None:
            return
        start, end = self._normalized_x_range((float(x_range[0]), float(x_range[1])))
        self._syncing_time_tf_x = True
        try:
            _, y_range = self.time_plot.getViewBox().viewRange()
            self._apply_view_state(((start, end), tuple(y_range)))
        finally:
            self._syncing_time_tf_x = False


    def _sync_feature_x_from_time(self, *_args) -> None:
        """Propagate the time plot's X range to Plot 2.
        """

        if self._syncing_time_feature_x or self._current_waveform is None:
            return
        x_range, _ = self.time_plot.getViewBox().viewRange()
        start, end = self._normalized_x_range((float(x_range[0]), float(x_range[1])))
        self._syncing_time_feature_x = True
        try:
            self.feature_plot.setXRange(start, end, padding=0.0)
        finally:
            self._syncing_time_feature_x = False


    def _sync_time_x_from_feature(self, _view_box, x_range) -> None:
        """Propagate Plot 2's X range to the time plot.
        """

        if self._syncing_time_feature_x or self._current_waveform is None:
            return
        start, end = self._normalized_x_range((float(x_range[0]), float(x_range[1])))
        self._syncing_time_feature_x = True
        try:
            _, y_range = self.time_plot.getViewBox().viewRange()
            self._apply_view_state(((start, end), tuple(y_range)))
        finally:
            self._syncing_time_feature_x = False


    def _clamp_time_plot_x_range(self) -> None:
        """Clamp a requested X range into the valid data span.

        Guarded by ``_clamping_time_x_range`` so the clamp cannot retrigger the
        synchronisation handlers that caused it.
        """

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
        """React to a user-driven change of the time plot's view.
        """

        self._clamp_time_plot_x_range()
        if self._fixed_psd_enabled:
            self._update_fixed_psd_window_from_view()


    def _update_fixed_psd_window_from_view(self) -> None:
        """Re-anchor the fixed PSD window when the view changes.
        """

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
        self.feature_plot.set_selection_region(start, end)
        self._fixed_psd_window_samples = max(1, end - start)
        self._update_psd_from_selection(start, end)


    def _update_psd_from_selection(self, start_index: int, end_index: int) -> None:
        """Compute and draw the PSD for the selected sample window.

        The PSD is always computed from the *unfiltered* signal, even when a display
        filter is active, so filtering never distorts the analysis.
        """

        if self._current_waveform is None:
            return

        self._clear_psd_plot()
        updated_labels: list[str] = []
        for channel_index in self._selected_psd_channel_indices():
            try:
                channel_values = self._current_waveform.channel_data(channel_index)
            except IndexError:
                continue

            bounded_start = max(0, min(int(start_index), channel_values.size))
            bounded_end = max(0, min(int(end_index), channel_values.size))
            if bounded_end <= bounded_start:
                bounded_end = min(channel_values.size, bounded_start + 1)
            if bounded_end <= bounded_start:
                continue

            raw_values = channel_values[bounded_start:bounded_end]
            freqs, psd_db = compute_window_psd(raw_values, self._current_waveform.sample_rate)
            valid = freqs > 0.0
            freqs = freqs[valid]
            psd_db = psd_db[valid]
            if freqs.size == 0:
                continue

            curve = self.psd_curve if channel_index == 0 else self.psd_curve_channel_2
            curve.setData(freqs, psd_db)
            updated_labels.append(self._current_waveform.channel_label(channel_index))

        if not updated_labels:
            self.statusBar().showMessage("Selection window is too short for PSD.")
            return

        psd_x_range_applied = self._apply_psd_x_range()
        self._apply_psd_y_range()
        window_seconds = max(0.0, (end_index - start_index) / self._current_waveform.sample_rate)
        self.window_length_label.setText(f"Window: {window_seconds:.6f} s")
        if psd_x_range_applied:
            self.statusBar().showMessage(
                f"PSD updated for {', '.join(updated_labels)} samples {start_index} to {end_index}."
            )


    def _record_view_history(self, _, view_range) -> None:
        """Push the current view onto the undo history, coalescing duplicates.
        """

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
        """Append the current view state to the history stack.
        """

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
        """Pop the history stack and apply the previous view.
        """

        if not self._view_history:
            return
        state = self._view_history.pop()
        self._apply_view_state(state)
        self.back_view_button.setEnabled(bool(self._view_history))
        self.statusBar().showMessage("Returned to previous view.")


    def _clear_view_history(self) -> None:
        """Empty the undo history.
        """

        self._view_history.clear()
        self._last_view_state = None
        self.back_view_button.setEnabled(False)


    def _current_view_state(self) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
        """Snapshot the current X/Y ranges of the time plot.
        """

        if self._current_waveform is None:
            return None
        x_range, y_range = self.time_plot.getViewBox().viewRange()
        return (
            (float(x_range[0]), float(x_range[1])),
            (float(y_range[0]), float(y_range[1])),
        )


    def _default_y_range(self) -> tuple[float, float]:
        """Return the auto Y range for the current data.
        """

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
        """Apply a stored X/Y view state, suppressing history recording.
        """

        self._suspend_history = True
        self._syncing_time_feature_x = True
        try:
            x_range, y_range = state
            x_range = self._normalized_x_range(x_range)
            self.time_plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
            self.time_plot.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
            self.time_plot.setXRange(x_range[0], x_range[1], padding=0.0)
            self.time_plot.getViewBox().setYRange(y_range[0], y_range[1], padding=0.0)
            self.feature_plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
            self.feature_plot.setXRange(x_range[0], x_range[1], padding=0.0)
        finally:
            self._suspend_history = False
            self._syncing_time_feature_x = False
        self._last_view_state = state
        self._refresh_length_labels()


    def _update_visible_length_label(self, *_args) -> None:
        """Refresh the visible-duration label.
        """

        self._refresh_length_labels()


    def _apply_visible_window_duration(self) -> None:
        """Set the visible span to the requested duration, centred on the view.
        """

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
        """Refresh the visible-duration and window-length labels together.
        """

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
        """Apply the PSD Y range in dB.
        """

        view_box = self.psd_plot.getViewBox()
        y_min = int(self.psd_y_min_spin.value())
        y_max = int(self.psd_y_max_spin.value())
        if y_min >= y_max:
            self.statusBar().showMessage("Invalid PSD Y range. Kept previous range.")
            return
        view_box.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
        view_box.setRange(yRange=(float(y_min), float(y_max)), padding=0.0, disableAutoRange=True)


    def _apply_psd_x_range(self) -> bool:
        """Apply the PSD X range, clamping to the file's Nyquist frequency.

        ``0 / 0`` means automatic; otherwise the values are applied in Hz.
        """

        if self._current_waveform is None:
            self._update_psd_x_axis_ticks()
            return False

        nyquist = max(float(self._current_waveform.sample_rate) * 0.5, 1.0)
        requested_min = float(self.psd_x_min_spin.value())
        requested_max = float(self.psd_x_max_spin.value())
        if requested_min == 0.0 and requested_max == 0.0:
            lower_hz = max(1.0, min(1000.0, nyquist))
            upper_hz = nyquist
        else:
            lower_hz = max(requested_min, 1e-12)
            upper_hz = requested_max if requested_max > 0.0 else nyquist
            upper_hz = min(upper_hz, nyquist)
            if lower_hz >= upper_hz:
                self.statusBar().showMessage("Invalid PSD X range. Kept previous range.")
                self._update_psd_x_axis_ticks()
                return False

        self.psd_plot.getViewBox().enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
        self.psd_plot.setXRange(np.log10(lower_hz), np.log10(upper_hz), padding=0.0)
        self._update_psd_x_axis_ticks()
        return True


    def _update_psd_x_axis_ticks(self, *_args) -> None:
        """Refresh the PSD log-frequency tick labels.

        When the visible span covers at least one decade only powers of ten are
        labelled, and the values carry no ``Hz`` suffix.
        """

        axis = self.psd_plot.getPlotItem().getAxis("bottom")
        x_min, x_max = sorted(float(value) for value in self.psd_plot.getViewBox().viewRange()[0])
        if x_max - x_min < 1.0:
            axis.setTicks(None)
            return

        first_decade = int(np.ceil(x_min))
        last_decade = int(np.floor(x_max))
        major_ticks: list[tuple[float, str]] = []
        minor_ticks: list[tuple[float, str]] = []
        for decade in range(first_decade, last_decade + 1):
            frequency = 10.0 ** decade
            if frequency >= 1.0:
                major_ticks.append((float(decade), f"{frequency:.0f}"))

        minor_start = int(np.floor(x_min))
        minor_stop = int(np.ceil(x_max))
        for decade in range(minor_start, minor_stop + 1):
            for factor in range(2, 10):
                tick = decade + float(np.log10(factor))
                if x_min <= tick <= x_max:
                    minor_ticks.append((tick, ""))

        axis.setTicks([major_ticks, minor_ticks])


    def _normalized_x_range(self, x_range: tuple[float, float]) -> tuple[float, float]:
        """Return the time plot's X range as a sorted ``(lo, hi)`` tuple.
        """

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
        """Sync the horizontal scrollbar with the time plot's view.

        Guarded by ``_syncing_scrollbar`` to avoid feedback between the two.
        """

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
        """Pan the time plot when the scrollbar moves.
        """

        if self._syncing_scrollbar or self._current_waveform is None:
            return
        x_range, y_range = self.time_plot.getViewBox().viewRange()
        visible_span = max(1.0, float(x_range[1]) - float(x_range[0]))
        self._apply_view_state(((float(value), float(value) + visible_span), tuple(y_range)))


    def _scroll_time_plot_by_step(self, direction: int) -> None:
        """Shift the visible window by one scrollbar step.
        """

        if self._current_waveform is None or self._current_display_values.size <= 1:
            return
        step = max(1, self.time_scrollbar.singleStep())
        new_value = self.time_scrollbar.value() + direction * step
        new_value = min(max(new_value, self.time_scrollbar.minimum()), self.time_scrollbar.maximum())
        if new_value != self.time_scrollbar.value():
            self.time_scrollbar.setValue(new_value)

