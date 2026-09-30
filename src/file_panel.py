"""Directory browsing, file list paging, threshold filtering and loading.

Part of the ``MainWindow`` mixin set; see ``main_window`` for the full table of
responsibilities.
"""

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from data_access import list_data_files
from data_access import load_waveform
from data_access import paginate_files
from models import FileRecord
from models import LoadedWaveform
from models import PAGE_SIZE
from pathlib import Path
from typing import Optional
from widgets import LoadWaveformWorker
import numpy as np


class FilePanelMixin:
    """Directory browsing, file-list paging, amplitude threshold filter and loading.

    Loading runs on a ``LoadWaveformWorker`` thread so a large directory or a slow
    disk never blocks the UI; a task id lets a newer request supersede an in-flight
    one.
    """

    def _handle_sample_type_combo_hover(self, index: QtCore.QModelIndex) -> None:
        """Show the sub-category tooltip for the hovered major sample type.
        """

        if not index.isValid():
            return
        text = self._sample_type_major_tooltips.get(int(index.row()))
        if text:
            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), text, self.sample_type_combo.view())


    def _set_sample_type_text(self, value: str) -> None:
        """Write a sample-type text into the combo box without firing handlers.
        """

        self.sample_type_combo.blockSignals(True)
        self.sample_type_combo.setEditText(value)
        self.sample_type_combo.blockSignals(False)


    def _current_sample_type_code(self) -> Optional[str]:
        """Return the currently entered sample-type code, or ``None`` if empty.
        """

        text = self.sample_type_combo.currentText().strip()
        if not text:
            return None
        return text.upper().replace(" ", "")


    def _sample_type_filename_token(self, sample_type: str) -> str:
        """Return the sample-type prefix used when building export names.
        """

        code = str(sample_type).strip().upper().replace(" ", "")
        safe = "".join(ch for ch in code if ch.isalnum())
        return safe if safe else "TAG"


    def _handle_sample_type_text_changed(self, text: str) -> None:
        """React to a manually typed sample-type code.
        """

        normalized = text.strip().upper().replace(" ", "")
        if normalized:
            self.statusBar().showMessage(f"Sample type set to {normalized}.")


    def _choose_directory(self) -> None:
        """Open a directory picker and reload the file list.
        """

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
        """Open a directory picker and remember it as the export target.
        """

        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory",
            self.export_directory_edit.text(),
        )
        if directory:
            self.export_directory_edit.setText(directory)


    def _refresh_file_list(self) -> None:
        """Rescan the current directory and repopinate the file list.
        """

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
        """Move to a page of the file list and refresh the pager controls.

        Rebuilds the list widget from the paginated slice of ``_all_files``, then
        updates the page label, the jump spin box and the enabled state of the
        first/prev/next/last buttons. The page index is clamped inside
        :func:`paginate_files`, so a stale index lands on the nearest valid page
        instead of raising.
        """

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
            self._update_channel_option_controls()
            self._current_display_values = np.array([], dtype=np.float64)
            self._audio_player.stop()
            self._clear_audio_temp_path()
            self.time_curve.setData([])
            self._clear_tf_time_plot()
            self._clear_short_time_feature_plot()
            self._clear_psd_plot()
            self._clear_time_frequency_plot()
            self._set_fixed_psd_enabled(False)
            self.time_plot.clear_selection_region()
            self.feature_plot.clear_selection_region()
            self.visible_length_label.setText("Visible: 0.000 s")
            self.window_length_label.setText("Window: 0.000 s")
            self._audio_path_auto_managed = True
            self.audio_path_edit.setText(str(self._default_audio_path()))
            self._update_time_scrollbar()


    def _go_to_last_page(self) -> None:
        """Jump to the final page of the file list.
        """

        page_count = paginate_files(self._all_files, 0, PAGE_SIZE).page_count
        self._change_page(max(0, page_count - 1))


    def _jump_to_page(self) -> None:
        """Jump to the page typed into the page-jump box.
        """

        if not self._all_files:
            return
        self._change_page(int(self.page_jump_spin.value()) - 1)


    def _handle_file_selection(self) -> None:
        """Start loading the files currently selected in the list.

        Supports Ctrl/Shift multi-selection; multi-file selections are handed to the
        concatenating loader.
        """

        paths: list[Path] = []
        for item in self.file_list.selectedItems():
            path = item.data(QtCore.Qt.UserRole)
            if path:
                paths.append(Path(path))
        if not paths:
            return

        self._start_waveform_load(paths)


    def _apply_amplitude_threshold_filter(self) -> None:
        """Filter the file list by peak-amplitude threshold.

        Reads each candidate file's peak amplitude, drops the ones below the
        threshold and re-paginates the survivors. A progress dialog and a wait
        cursor are shown because the scan reads the header of every file; a
        progress dialog with no cancel button is used rather than a worker
        thread so the partial result is never applied to a stale file list.
        """

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


    def _start_waveform_load(self, paths: list[Path]) -> None:
        """Kick off a background load for the given paths.

        Starts a ``QThread`` running ``LoadWaveformWorker`` and records the task id, so
        a newer request can supersede an in-flight one.
        """

        if not paths:
            return
        self._load_task_id += 1
        task_id = self._load_task_id
        self._prediction_task_id += 1
        self._current_waveform = None
        self._set_sample_type_text("")
        self._clear_arrival_marker()
        self._current_display_values = np.array([], dtype=np.float64)
        self._audio_player.stop()
        self._clear_audio_temp_path()
        self.time_curve.setData([])
        self._clear_tf_time_plot()
        self._clear_short_time_feature_plot()
        self._clear_psd_plot()
        self._clear_time_frequency_plot()
        self._set_fixed_psd_enabled(False)
        self.time_plot.clear_selection_region()
        self.visible_length_label.setText("Visible: 0.000 s")
        self.window_length_label.setText("Window: 0.000 s")
        self._update_time_scrollbar()
        if len(paths) == 1:
            self.statusBar().showMessage(f"Loading {paths[0].name}...")
        else:
            self.statusBar().showMessage(f"Loading and concatenating {len(paths)} files by start time...")

        thread = QtCore.QThread(self)
        worker = LoadWaveformWorker(task_id, paths)
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
        """Tear down the loader thread and worker once loading ends.
        """

        self._load_thread = None
        self._load_worker = None


    @QtCore.pyqtSlot(int, object)
    def _handle_waveform_loaded(self, task_id: int, waveform: LoadedWaveform) -> None:
        """Adopt a successfully loaded waveform and rebuild the plots.
        """

        if task_id != self._load_task_id:
            return
        self._current_waveform = waveform
        self._update_channel_option_controls()
        self._set_sample_type_text(waveform.sample_type or "")
        self._rebuild_time_plot()
        self._apply_loaded_arrival_time(waveform)
        header_message = self._build_waveform_header_message(waveform)
        warning = waveform.data_info_warning
        if warning:
            self.statusBar().showMessage(f"{header_message} | warning: {warning}")
        else:
            self.statusBar().showMessage(header_message)


    @QtCore.pyqtSlot(int, str)
    def _handle_waveform_load_failed(self, task_id: int, message: str) -> None:
        """Report a load failure in the status bar.
        """

        if task_id != self._load_task_id:
            return
        self._current_waveform = None
        self._update_channel_option_controls()
        self._clear_arrival_marker()
        self._current_display_values = np.array([], dtype=np.float64)
        self.time_curve.setData([])
        self._clear_tf_time_plot()
        self._clear_short_time_feature_plot()
        self._clear_psd_plot()
        self._clear_time_frequency_plot()
        self._set_fixed_psd_enabled(False)
        self.visible_length_label.setText("Visible: 0.000 s")
        self.window_length_label.setText("Window: 0.000 s")
        self._update_time_scrollbar()
        self.statusBar().showMessage(f"Failed to open file: {message}")


    def _build_waveform_header_message(self, waveform: LoadedWaveform) -> str:
        """Build the status-bar summary of the loaded file.

        Reports sample rate, arrival time, duration and sample type, as required by the
        "show file header information" requirement.
        """

        sample_rate = float(waveform.sample_rate)
        duration = float(waveform.phase_data.size) / max(sample_rate, 1.0)
        arrival_text = self._format_arrival_time(waveform.arrival_time)
        sample_type_text = waveform.sample_type if waveform.sample_type else "None"
        channel_text = f"channels={waveform.channel_count}"
        if waveform.channel_count > 1:
            channel_text = (
                f"channels={waveform.channel_count} "
                f"({', '.join(waveform.channel_label(index) for index in range(waveform.channel_count))})"
            )
        return (
            f"{waveform.path.name} | sample_rate={sample_rate:g} Hz | "
            f"{channel_text} | sample_type={sample_type_text} | "
            f"arrival_time={arrival_text} | duration={duration:.6f} s"
        )

