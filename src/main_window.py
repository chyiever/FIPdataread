"""Composed FIPread main window.

``MainWindow`` itself is intentionally small: it holds only the constructor that
establishes the shared runtime state, and it composes the behaviour mixins that
hold the actual logic. The mixins are plain classes with no ``__init__`` and no
``super()`` calls, so method resolution and every ``self.*`` reference behave
exactly as they did when this logic lived in one 3500-line class.

Behaviour mixins
----------------
===============================  ==========================================
Module                           Responsibility
===============================  ==========================================
``ui_builder``                   Widget tree construction, fonts, theme,
                                 signal binding
``file_panel``                   Directory pick, file list, paging,
                                 amplitude filter, waveform loading,
                                 sample-type tags
``plot_panel``                   Time-domain plot, Welch PSD, interaction
                                 modes, view history, X-axis sync
``time_frequency_panel``         t-f Plot tab: spectrogram, log-frequency
                                 display grid, colormap and colour levels
``short_time_feature_panel``     Plot 2 short-time features and the
                                 sliding-window SVM prediction worker
``audio_panel``                  Visible-segment playback and WAV export
``export_panel``                 Raw visible-data export and first-arrival
                                 (``arrival_time``) marking
``layout``                       Splitter sizing, t-f tab alignment, window
                                 resize handling
===============================  ==========================================

Supporting modules
------------------
``constants``  Feature-mode / PSD-source / t-f identifier strings.
``resources``  Resolves the resource root for source runs and PyInstaller.
``widgets``    Background workers and the ``TimePlotWidget`` plot class.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtMultimedia, QtWidgets

from audio_panel import AudioPanelMixin
from config import UI_DEFAULTS
from constants import (
    FEATURE_MODE_BAND_ENERGY,
    FEATURE_MODE_CHANNEL_1,
    FEATURE_MODE_CHANNEL_2,
    FEATURE_MODE_ENERGY,
    FEATURE_MODE_ENERGY_ENERGY,
    FEATURE_MODE_MAX_NUM,
    FEATURE_MODE_NONE,
    FEATURE_MODE_PSD_SUM,
    FEATURE_MODE_SVM,
    PSD_SOURCE_BOTH,
    PSD_SOURCE_CHANNEL_1,
    PSD_SOURCE_CHANNEL_2,
    TF_MODE_AMPLITUDE,
    TF_MODE_PSD,
    TF_SCALE_LINEAR,
    TF_SCALE_LOG,
    TF_SOURCE_CHANNEL_1,
    TF_SOURCE_CHANNEL_2,
)
from export_panel import ExportPanelMixin
from file_panel import FilePanelMixin
from layout import LayoutMixin
from models import FileRecord, LoadedWaveform
from plot_panel import PlotPanelMixin
from resources import APPLICATION_ROOT, application_root
from short_time_feature_panel import ShortTimeFeaturePanelMixin
from time_frequency_panel import TimeFrequencyPanelMixin
from ui_builder import UiBuilderMixin
from widgets import LoadWaveformWorker, SVMPredictionWorker, TimePlotWidget

__all__ = [
    "MainWindow",
    "LoadWaveformWorker",
    "SVMPredictionWorker",
    "TimePlotWidget",
    "APPLICATION_ROOT",
    "application_root",
    "FEATURE_MODE_NONE",
    "FEATURE_MODE_CHANNEL_1",
    "FEATURE_MODE_CHANNEL_2",
    "FEATURE_MODE_SVM",
    "FEATURE_MODE_ENERGY",
    "FEATURE_MODE_BAND_ENERGY",
    "FEATURE_MODE_ENERGY_ENERGY",
    "FEATURE_MODE_PSD_SUM",
    "FEATURE_MODE_MAX_NUM",
    "PSD_SOURCE_CHANNEL_1",
    "PSD_SOURCE_CHANNEL_2",
    "PSD_SOURCE_BOTH",
    "TF_SOURCE_CHANNEL_1",
    "TF_SOURCE_CHANNEL_2",
    "TF_MODE_PSD",
    "TF_MODE_AMPLITUDE",
    "TF_SCALE_LOG",
    "TF_SCALE_LINEAR",
]


class MainWindow(
    UiBuilderMixin,
    FilePanelMixin,
    PlotPanelMixin,
    TimeFrequencyPanelMixin,
    ShortTimeFeaturePanelMixin,
    AudioPanelMixin,
    ExportPanelMixin,
    LayoutMixin,
    QtWidgets.QMainWindow,
):
    """The application main window.

    Composes the behaviour mixins listed in this module's docstring and owns
    only the shared runtime state established in ``__init__``.
    """

    def __init__(self) -> None:
        """Create the window and establish all shared runtime state.

        State is initialised before ``_build_ui`` runs because the theme/font helpers
        read the defaults from ``UI_DEFAULTS``, and the plotting helpers need the
        history/guard flags below to be present while the first plots are created.

        Guarded re-entrancy flags:

        * ``_suspend_history`` - suppresses view-history recording during programmatic
          view changes so restoring a view does not push a new history entry.
        * ``_syncing_scrollbar`` / ``_syncing_time_tf_x`` / ``_syncing_time_feature_x`` /
          ``clamping_time_x_range`` - break feedback loops between the scrollbar, the t-f
          plot, Plot 2 and the time plot.
        * ``_updating_tf_color_spins`` - blocks the colour-spin handlers while their
          values are being written programmatically.
        """

        super().__init__()
        self.setWindowTitle("FIPread")
        self._apply_initial_window_size()

        self._source_files: list[FileRecord] = []
        self._all_files: list[FileRecord] = []
        self._page_index = 0
        self._current_waveform: Optional[LoadedWaveform] = None
        self._current_display_values = np.array([], dtype=np.float64)
        self._tf_time_display_values = np.array([], dtype=np.float64)
        self._view_history: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self._last_view_state: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
        self._suspend_history = False
        self._syncing_scrollbar = False
        self._scroll_shortcuts: list[QtWidgets.QShortcut] = []
        self._fixed_psd_enabled = False
        self._fixed_psd_anchor_ratio = 0.5
        self._fixed_psd_window_samples = 0
        self._svm_model_directory = APPLICATION_ROOT / "models" / "saved_models"
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
        self._syncing_time_feature_x = False
        self._tf_freq_hz = np.array([], dtype=np.float64)
        self._tf_time_centers = np.array([], dtype=np.float64)
        self._tf_base_values = np.array([], dtype=np.float64)
        self._tf_base_mode = TF_MODE_PSD
        self._tf_log_freq_bounds: Optional[tuple[float, float]] = None
        self._tf_default_color_min = UI_DEFAULTS.display.tf_color_min
        self._tf_color_min_user_override = False
        self._updating_tf_color_spins = False
        self._clamping_time_x_range = False
        self._tf_side_panel_width = 88
        self._tf_time_axis_compensation = 102
        self._arrival_line: Optional[pg.InfiniteLine] = None
        self._arrival_sample_index: Optional[float] = None
        self._arrival_time: Optional[datetime] = None
        self._sample_type_major_tooltips: dict[int, str] = {}

        self._build_ui()
        self._bind_events()
        self._refresh_file_list()

