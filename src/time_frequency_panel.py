"""The t-f Plot tab: spectrogram, log-frequency grid and colormaps.

Part of the ``MainWindow`` mixin set; see ``main_window`` for the full table of
responsibilities.
"""

from PyQt5 import QtCore
from constants import TF_MODE_PSD
from constants import TF_SCALE_LOG
from constants import TF_SOURCE_CHANNEL_1
from constants import TF_SOURCE_CHANNEL_2
from plotting import AbsoluteTimeAxis
from plotting import create_colormap
from processing import compute_time_frequency_map
from typing import Optional
import numpy as np
import pyqtgraph as pg


class TimeFrequencyPanelMixin:
    """The t-f Plot tab: spectrogram computation, log-frequency grid and colormaps.

    The spectrogram is resampled onto an evenly spaced ``log10(f)`` grid before it is
    handed to ``ImageItem``, because that item can only draw uniformly spaced rows
    and columns.
    """

    def _current_tf_source(self) -> str:
        """Return the selected t-f channel source identifier.
        """

        return str(self.tf_source_combo.currentData() or TF_SOURCE_CHANNEL_1)


    def _current_tf_channel_index(self) -> int:
        """Map the t-f source identifier to a channel index.
        """

        if self._current_tf_source() == TF_SOURCE_CHANNEL_2 and self._has_channel_2():
            return 1
        return 0


    def _handle_tf_source_changed(self, _index: int) -> None:
        """Recompute the t-f map after the CH selector changes.
        """

        self._refresh_time_plot_channel()
        self._rebuild_time_frequency_plot()


    def _handle_tf_color_auto_toggled(self, checked: bool) -> None:
        """Switch the colour level between automatic and manual.
        """

        manual_enabled = not bool(checked)
        self.tf_color_min_spin.setEnabled(manual_enabled)
        self.tf_color_max_spin.setEnabled(manual_enabled)
        self._apply_time_frequency_color_levels()


    def _handle_tf_color_min_changed(self, value: float) -> None:
        """Apply a manually entered minimum colour level.
        """

        if self._updating_tf_color_spins:
            return
        self._tf_color_min_user_override = abs(float(value) - self._tf_default_color_min) > 1e-9
        if self.tf_color_auto_checkbox.isChecked():
            self._apply_time_frequency_color_levels()


    def _lookup_tf_cursor_value(self, sample_index: float, freq_hz: float) -> Optional[tuple[float, str]]:
        """Return the t-f value under the cursor as ``(time_s, value)``.
        """

        if self._tf_time_centers.size == 0 or self._tf_freq_hz.size == 0 or self._tf_base_values.size == 0:
            return None
        valid = self._tf_freq_hz > 0.0
        if not np.any(valid):
            return None
        freqs = self._tf_freq_hz[valid]
        values = self._tf_base_values[valid, :]
        if values.size == 0:
            return None
        time_idx = int(np.argmin(np.abs(self._tf_time_centers - sample_index)))
        freq_idx = int(np.argmin(np.abs(freqs - freq_hz)))
        base_value = float(values[freq_idx, time_idx])
        if self._current_tf_scale() == TF_SCALE_LOG:
            floor = np.finfo(np.float64).tiny
            if self._tf_base_mode == TF_MODE_PSD:
                return 10.0 * np.log10(max(base_value, floor)), "dB(PSD)"
            return 20.0 * np.log10(max(base_value, floor)), "dB(Amp)"
        return base_value, "linear"


    def _handle_tf_plot_mouse_moved(self, scene_pos: QtCore.QPointF) -> None:
        """Update the t-f cursor readout while the mouse moves.
        """

        if self._current_waveform is None:
            return
        view_box = self.tf_plot.getViewBox()
        if not view_box.sceneBoundingRect().contains(scene_pos):
            return
        point = view_box.mapSceneToView(scene_pos)
        x = float(point.x())
        y_log = float(point.y())
        freq_hz = 10.0 ** y_log
        time_text = self._format_cursor_time(x)
        if time_text is None:
            return
        tf_value = self._lookup_tf_cursor_value(x, freq_hz)
        if tf_value is None:
            self.statusBar().showMessage(f"Cursor(t-f): t={time_text}, f={freq_hz:.3f} Hz")
            return
        value, unit = tf_value
        self.statusBar().showMessage(
            f"Cursor(t-f): t={time_text}, f={freq_hz:.3f} Hz, value={value:.6g} {unit}"
        )


    def _build_tf_display_values(self) -> Optional[np.ndarray]:
        """Return the display-filtered data for the t-f channel.
        """

        if self._current_waveform is None:
            return None
        return self._build_channel_display_values(
            self._current_waveform,
            self._current_tf_channel_index(),
        )


    def _rebuild_tf_time_plot(self) -> None:
        """Redraw the compact time plot shown above the t-f image.
        """

        if self._current_waveform is None:
            self._clear_tf_time_plot()
            return

        values = self._build_tf_display_values()
        if values is None:
            self._clear_tf_time_plot()
            return

        self._tf_time_display_values = np.asarray(values, dtype=np.float64)
        self.tf_time_curve.setData(self._tf_time_display_values)
        bottom_axis = self.tf_time_plot.getPlotItem().getAxis("bottom")
        if isinstance(bottom_axis, AbsoluteTimeAxis):
            bottom_axis.set_context(
                start_time=self._current_waveform.start_time,
                sample_rate=self._current_waveform.sample_rate,
            )
        if self._tf_time_display_values.size > 1:
            x_range, _ = self.time_plot.getViewBox().viewRange()
            start, end = self._normalized_x_range((float(x_range[0]), float(x_range[1])))
            self.tf_time_plot.setXRange(start, end, padding=0.0)
        self.tf_time_plot.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)


    def _clear_tf_time_plot(self) -> None:
        """Clear the compact t-f time plot.
        """

        self._tf_time_display_values = np.array([], dtype=np.float64)
        self.tf_time_curve.setData([])


    def _apply_time_frequency_params(self) -> None:
        """Re-apply every t-f parameter (recomputes the map).
        """

        self._rebuild_time_frequency_plot()


    def _current_tf_mode(self) -> str:
        """Return the selected spectrogram mode.
        """

        return str(self.tf_mode_combo.currentData())


    def _current_tf_scale(self) -> str:
        """Return the selected t-f value scale.
        """

        return str(self.tf_value_scale_combo.currentData())


    def _current_tf_colormap_name(self) -> str:
        """Return the selected colormap name.
        """

        return self.tf_colormap_combo.currentText().strip() or "jet"


    def _clear_time_frequency_plot(self) -> None:
        """Drop the cached t-f data, blank the image and reset the axis ticks.

        The cached frequency, time and value arrays are emptied so a later rebuild
        knows there is nothing to reuse, and the left axis falls back to pyqtgraph's
        default tick behaviour.
        """

        self._tf_freq_hz = np.array([], dtype=np.float64)
        self._tf_time_centers = np.array([], dtype=np.float64)
        self._tf_base_values = np.array([], dtype=np.float64)
        self._tf_log_freq_bounds = None
        self.tf_image_item.setImage(np.empty((0, 0), dtype=np.float64), autoLevels=False)
        self.tf_plot.getPlotItem().getAxis("left").setTicks(None)


    def _update_time_frequency_axis_ticks(self) -> None:
        """Refresh the t-f log-frequency and time tick labels.

        Minor ticks use standard base-10 positions (``2..9 x 10^n``) rather than equal
        subdivisions, matching the fix documented in the 2026-04-16 log entry.
        """

        axis = self.tf_plot.getPlotItem().getAxis("left")
        if self._tf_log_freq_bounds is None:
            axis.setTicks(None)
            return

        y_range = self.tf_plot.getViewBox().viewRange()[1]
        y_min, y_max = sorted((float(y_range[0]), float(y_range[1])))
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
            return

        lo_decade = int(np.floor(y_min))
        hi_decade = int(np.ceil(y_max))
        major_ticks: list[tuple[float, str]] = []
        minor_ticks: list[tuple[float, str]] = []

        for decade in range(lo_decade, hi_decade + 1):
            tick = float(decade)
            if y_min <= tick <= y_max:
                freq = 10.0 ** tick
                if freq >= 10000.0:
                    label = f"{freq:.0f}"
                elif freq >= 1000.0:
                    label = f"{freq:.1f}"
                elif freq >= 10.0:
                    label = f"{freq:.0f}"
                else:
                    label = f"{freq:.2f}"
                major_ticks.append((tick, label))
            # Standard base-10 log subticks: 2..9 within each decade.
            for factor in range(2, 10):
                sub_tick = decade + float(np.log10(factor))
                if y_min <= sub_tick <= y_max:
                    minor_ticks.append((sub_tick, ""))

        axis.setTicks([major_ticks, minor_ticks])


    def _handle_tf_y_range_changed(self, *_args) -> None:
        """Re-apply the t-f frequency range after an edit.
        """

        if self._tf_log_freq_bounds is None:
            return
        self._update_time_frequency_axis_ticks()


    def _rebuild_time_frequency_plot(self) -> None:
        """Recompute and redraw the whole t-f view.

        Resamples the spectrogram onto the display grid, renders the image, then applies
        the colormap, Y range and colour levels.
        """

        if self._current_waveform is None:
            self._clear_time_frequency_plot()
            return
        source_values = self._build_tf_display_values()
        if source_values is None:
            self._clear_time_frequency_plot()
            return
        source_values = np.asarray(source_values, dtype=np.float64)
        if source_values.size == 0:
            self._clear_time_frequency_plot()
            return
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
        """Draw the t-f image and apply the ranges, colour levels and X sync.

        The image rectangle is built from the median time and log-frequency bin
        spacing, extended by half a bin on each side so the outermost pixels are
        centred on the first and last bin instead of being clipped. The Y limits
        leave a 20% margin above the top bin so the highest decade stays visible.
        Horizontal frequency grid lines are deliberately disabled in ``_build_ui``
        (``showGrid(x=True, y=False)``) so they cannot be confused with the short
        axis ticks.
        """

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
        display_values, log_freq = self._build_time_frequency_display_grid()
        if display_values.size == 0 or log_freq.size == 0:
            self._clear_time_frequency_plot()
            self.statusBar().showMessage("t-f plot could not build a valid display grid.")
            return

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
            yMax=float(log_freq[-1]) + (1.2 * dy),
        )
        self._apply_time_frequency_y_range()
        self._apply_time_frequency_color_levels(display_values)
        self._sync_tf_x_from_time()
        mode_label = "PSD" if self._tf_base_mode == TF_MODE_PSD else "Amplitude"
        channel_label = self._current_waveform.channel_label(self._current_tf_channel_index())
        self.statusBar().showMessage(
            f"t-f plot updated: {channel_label}, {mode_label}, "
            f"{self._tf_time_centers.size} windows, {freqs.size} frequency bins."
        )


    def _build_time_frequency_display_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Resample the linear-frequency spectrogram onto a log10(f) grid.

        ``ImageItem`` only supports uniformly spaced rows and columns, so a
        linear-frequency matrix cannot be drawn directly on a log-frequency axis. Each
        column is interpolated onto an evenly spaced ``log10(f)`` grid, which fixes the
        bug where high-frequency energy appeared at too low a displayed frequency. The
        same grid is reused for colour-level computation to keep image and histogram
        consistent.
        """

        valid = self._tf_freq_hz > 0.0
        if not np.any(valid):
            return np.empty((0, 0), dtype=np.float64), np.array([], dtype=np.float64)

        freqs = np.asarray(self._tf_freq_hz[valid], dtype=np.float64)
        base_values = np.asarray(self._tf_base_values[valid, :], dtype=np.float64)
        if freqs.size == 0 or base_values.size == 0:
            return np.empty((0, 0), dtype=np.float64), np.array([], dtype=np.float64)

        if freqs.size == 1:
            log_freq = np.array([float(np.log10(freqs[0]))], dtype=np.float64)
            resampled_values = base_values.copy()
        else:
            # Evenly spaced grid in log10(f), which is what the log-frequency
            # axis expects. Each output row is the linear interpolation between
            # the two neighbouring source frequencies, so no energy is shifted
            # to the wrong decade.
            log_freq = np.linspace(float(np.log10(freqs[0])), float(np.log10(freqs[-1])), freqs.size, dtype=np.float64)
            target_freqs = np.power(10.0, log_freq)
            # searchsorted locates the bracketing source bins for every target
            # frequency; clamping keeps the last target inside the valid range.
            lower_indices = np.searchsorted(freqs, target_freqs, side="right") - 1
            lower_indices = np.clip(lower_indices, 0, freqs.size - 2)
            x0 = freqs[lower_indices]
            x1 = freqs[lower_indices + 1]
            spacing = np.maximum(x1 - x0, np.finfo(np.float64).eps)
            weights = ((target_freqs - x0) / spacing).astype(np.float64)
            resampled_values = (
                base_values[lower_indices, :] * (1.0 - weights[:, None])
                + base_values[lower_indices + 1, :] * weights[:, None]
            )

        # The dB conversion is done on the resampled grid, not before it, so the
        # image and the colour-level histogram see identical values.
        if self._current_tf_scale() == TF_SCALE_LOG:
            floor = np.finfo(np.float64).tiny
            if self._tf_base_mode == TF_MODE_PSD:
                display_values = 10.0 * np.log10(np.maximum(resampled_values, floor))
            else:
                display_values = 20.0 * np.log10(np.maximum(resampled_values, floor))
        else:
            display_values = resampled_values
        return np.asarray(display_values, dtype=np.float64), log_freq


    def _apply_time_frequency_colormap(self) -> None:
        """Apply the selected colormap to the t-f view.
        """

        color_map = create_colormap(self._current_tf_colormap_name())
        self.tf_image_item.setColorMap(color_map)
        self.tf_histogram.item.gradient.setColorMap(color_map)


    def _apply_time_frequency_y_range(self) -> None:
        """Apply the t-f frequency Y range in Hz.
        """

        if self._tf_log_freq_bounds is None:
            return
        view_box = self.tf_plot.getViewBox()
        lower_bound, upper_bound = self._tf_log_freq_bounds
        y_min = float(self.tf_y_min_spin.value())
        y_max = float(self.tf_y_max_spin.value())
        if y_min == 0.0 and y_max == 0.0:
            view_box.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
            top_margin = max((upper_bound - lower_bound) * 0.02, 0.02)
            view_box.setYRange(lower_bound, upper_bound + top_margin, padding=0.0)
            self._update_time_frequency_axis_ticks()
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
        top_margin = max((float(log_max) - float(log_min)) * 0.02, 0.02)
        view_box.setYRange(float(log_min), float(log_max) + top_margin, padding=0.0)
        self._update_time_frequency_axis_ticks()


    def _apply_time_frequency_color_levels(self, values: Optional[np.ndarray] = None) -> None:
        """Set the t-f colour levels, automatically or manually.

        Guarded by ``_updating_tf_color_spins`` so writing the spin boxes does not
        re-enter the handlers that changed them.
        """

        if values is None:
            if self._tf_base_values.size == 0:
                return
            values, _log_freq = self._build_time_frequency_display_grid()
            if values.size == 0:
                return
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

