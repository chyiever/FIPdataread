"""Raw visible-data export and first-arrival (arrival_time) marking.

Part of the ``MainWindow`` mixin set; see ``main_window`` for the full table of
responsibilities.
"""

from data_access import build_export_npz_name
from data_access import build_export_tdms_name
from data_access import build_export_txt_name
from data_access import format_arrival_time_token
from data_access import save_npz_waveform
from data_access import save_tdms_waveform
from data_access import save_txt_waveform
from datetime import datetime
from datetime import timedelta
from models import LoadedWaveform
from pathlib import Path
from plotting import make_pen
from typing import Optional
import numpy as np
import pyqtgraph as pg


class ExportPanelMixin:
    """Raw visible-data export and first-arrival (``arrival_time``) marking.

    Export supports NPZ, TDMS and TXT and keeps every loaded channel, not just CH1.
    The arrival marker stores both the sample index and the derived absolute
    datetime, so it survives an export/import round trip.
    """

    def _export_visible_raw_data(self) -> None:
        """Export the currently visible raw waveform.

        Supports NPZ, TDMS and TXT. The exported name honours the marked sample type
        (``BK-FIP-...``) and, for NPZ, embeds ``arrival_time`` and the sample ``type``
        code. All loaded channels are retained in the export, not just CH1.
        """

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
        channel_sample_count = min(channel.size for channel in self._current_waveform.channels)
        end_index = min(channel_sample_count, int(np.ceil(end_x)))
        if end_index <= start_index:
            end_index = min(channel_sample_count, start_index + 1)
        if end_index <= start_index:
            self.statusBar().showMessage('Visible window is empty.')
            return

        segment_channels = tuple(channel[start_index:end_index] for channel in self._current_waveform.channels)
        segment = segment_channels[0] if len(segment_channels) == 1 else segment_channels
        segment_start_time = self._current_waveform.start_time + timedelta(
            seconds=start_index / self._current_waveform.sample_rate
        )
        sample_type_code = self._current_sample_type_code()
        export_format = str(self.export_format_combo.currentData() or "npz").lower()
        if export_format == "npz":
            base_name = build_export_npz_name(segment_start_time, self._current_waveform.sample_rate)
            if sample_type_code is None:
                filename = base_name
            else:
                filename = f"{self._sample_type_filename_token(sample_type_code)}-{base_name}"
        elif export_format == "tdms":
            filename = build_export_tdms_name(segment_start_time, self._current_waveform.sample_rate)
        elif export_format == "txt":
            filename = build_export_txt_name(segment_start_time, self._current_waveform.sample_rate)
        else:
            self.statusBar().showMessage(f'Unsupported export format: {export_format}')
            return
        destination = export_directory / filename

        try:
            if export_format == "npz":
                save_npz_waveform(
                    destination,
                    segment,
                    self._current_waveform.sample_rate,
                    segment_start_time,
                    self._arrival_time,
                    sample_type_code,
                )
            elif export_format == "tdms":
                save_tdms_waveform(
                    destination,
                    segment,
                    self._current_waveform.sample_rate,
                    segment_start_time,
                )
            elif export_format == "txt":
                save_txt_waveform(destination, segment)
        except Exception as exc:
            self.statusBar().showMessage(f'Export failed: {exc}')
            return

        self.statusBar().showMessage(
            f'Exported visible raw data ({export_format.upper()}) to {destination}'
        )


    def _format_arrival_time(self, value: Optional[datetime]) -> str:
        """Format a marked arrival time, or a placeholder when unset.
        """

        if value is None:
            return "None"
        return format_arrival_time_token(value)


    def _arrival_datetime_from_sample(self, sample_index: float) -> Optional[datetime]:
        """Convert a marked sample index into an absolute datetime.
        """

        if self._current_waveform is None:
            return None
        sample_rate = max(float(self._current_waveform.sample_rate), 1.0)
        return self._current_waveform.start_time + timedelta(seconds=float(sample_index) / sample_rate)


    def _set_arrival_marker(self, sample_index: float, *, announce: bool) -> None:
        """Place (or move) the blue first-arrival line.

        Stores the sample index and the derived datetime, and updates the status bar.
        """

        if self._current_waveform is None or self._current_display_values.size == 0:
            self._clear_arrival_marker()
            return
        bounded = min(max(float(sample_index), 0.0), float(self._current_display_values.size - 1))
        if self._arrival_line is None:
            self._arrival_line = pg.InfiniteLine(
                pos=bounded,
                angle=90,
                pen=make_pen("#0066FF", 2),
                movable=False,
            )
            self.time_plot.addItem(self._arrival_line)
        else:
            self._arrival_line.setPos(bounded)
        self._arrival_sample_index = bounded
        self._arrival_time = self._arrival_datetime_from_sample(bounded)
        if announce and self._arrival_time is not None:
            self.statusBar().showMessage(f"First arrival time: {self._format_arrival_time(self._arrival_time)}")


    def _clear_arrival_marker(self) -> None:
        """Remove the first-arrival line and clear the stored time.
        """

        if self._arrival_line is not None:
            try:
                self.time_plot.removeItem(self._arrival_line)
            except Exception:
                pass
        self._arrival_line = None
        self._arrival_sample_index = None
        self._arrival_time = None


    def _mark_arrival_at_index(self, sample_index: float) -> None:
        """Mark the first arrival at a given sample index.
        """

        if self._current_waveform is None:
            return
        self._set_arrival_marker(sample_index, announce=True)


    def _move_arrival_marker(self, direction: int) -> None:
        """Nudge the first-arrival marker by a fixed number of samples.

        The step is defined in seconds and converted to samples, so the marker
        moves by the same *time* regardless of the file's sample rate.
        """

        if self._current_waveform is None:
            return
        if self._arrival_sample_index is None:
            self.statusBar().showMessage("First arrival marker is not set.")
            return
        step_seconds = 0.0001
        step_samples = step_seconds * max(float(self._current_waveform.sample_rate), 1.0)
        self._set_arrival_marker(self._arrival_sample_index + direction * step_samples, announce=True)


    def _apply_loaded_arrival_time(self, waveform: LoadedWaveform) -> None:
        """Restore a first-arrival marker from a loaded file.
        """

        if waveform.arrival_time is None:
            self._clear_arrival_marker()
            return
        delta_seconds = (waveform.arrival_time - waveform.start_time).total_seconds()
        sample_index = delta_seconds * max(float(waveform.sample_rate), 1.0)
        self._set_arrival_marker(sample_index, announce=False)

