"""Visible-segment audio playback, WAV export and media temp files.

Part of the ``MainWindow`` mixin set; see ``main_window`` for the full table of
responsibilities.
"""

from PyQt5 import QtCore
from PyQt5 import QtMultimedia
from PyQt5 import QtWidgets
from data_access import save_wav_waveform
from datetime import timedelta
from pathlib import Path
from processing import prepare_audio_waveform
from typing import Optional
import numpy as np
import tempfile


class AudioPanelMixin:
    """Visible-segment audio playback and WAV export.

    Playback always covers only the currently visible time-domain segment, which is
    written to a temporary WAV file and handed to ``QMediaPlayer``. The temp file is
    recreated whenever the view changes and deleted again on close.
    """

    def _refresh_default_audio_path(self) -> None:
        """Reset the audio path to its auto-managed default.
        """

        if self._audio_path_auto_managed:
            self.audio_path_edit.setText(str(self._default_audio_path()))


    def _choose_audio_path(self) -> None:
        """Open a file picker for a user-supplied audio file.
        """

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
        """Mark the audio path as manually managed when edited.
        """

        self._audio_path_auto_managed = False


    def _default_audio_path(self) -> Path:
        """Return the auto-managed temp path for exported audio.
        """

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


    def _get_visible_display_segment(self) -> Optional[tuple[np.ndarray, float, int, int]]:
        """Return the currently visible time-domain segment as samples.
        """

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
        """Convert the visible segment to 16-bit PCM for playback.
        """

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
        """Delete the temporary WAV file created for playback.
        """

        if self._audio_temp_path is not None and self._audio_temp_path.exists():
            try:
                self._audio_temp_path.unlink()
            except OSError:
                pass
        self._audio_temp_path = None


    def _prepare_visible_audio_media(self) -> Optional[tuple[int, int, int]]:
        """Write the visible segment to a temp WAV and return its path.
        """

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
        """Start or resume playback of the visible segment.
        """

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
        """Stop playback and release the media player.
        """

        if self._audio_player.state() == QtMultimedia.QMediaPlayer.StoppedState:
            self.statusBar().showMessage('Audio is not playing.')
            return
        self._audio_player.pause()
        self.statusBar().showMessage('Audio playback stopped.')


    def _replay_visible_audio(self) -> None:
        """Stop and replay the visible segment from the start.
        """

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
        """Write the visible segment to a user-chosen WAV file.
        """

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

