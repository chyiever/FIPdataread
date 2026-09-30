"""Construction and styling of the main window's widget tree.

Part of the ``MainWindow`` mixin set; see ``main_window`` for the full table of
responsibilities.
"""

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from config import UI_DEFAULTS
from constants import FEATURE_MODE_BAND_ENERGY
from constants import FEATURE_MODE_CHANNEL_1
from constants import FEATURE_MODE_CHANNEL_2
from constants import FEATURE_MODE_ENERGY
from constants import FEATURE_MODE_ENERGY_ENERGY
from constants import FEATURE_MODE_MAX_NUM
from constants import FEATURE_MODE_NONE
from constants import FEATURE_MODE_PSD_SUM
from constants import FEATURE_MODE_SVM
from constants import PSD_SOURCE_BOTH
from constants import PSD_SOURCE_CHANNEL_1
from constants import PSD_SOURCE_CHANNEL_2
from constants import TF_MODE_AMPLITUDE
from constants import TF_MODE_PSD
from constants import TF_SCALE_LINEAR
from constants import TF_SCALE_LOG
from constants import TF_SOURCE_CHANNEL_1
from constants import TF_SOURCE_CHANNEL_2
from models import FilterMode
from models import InteractionMode
from models import PAGE_SIZE
from models import SortField
from pathlib import Path
from plotting import AXIS_TICK_FONT_SIZE_PT
from plotting import AbsoluteTimeAxis
from plotting import LogFrequencyAxis
from plotting import LogPowerFrequencyAxis
from plotting import configure_plot_widget
from plotting import make_pen
from resources import APPLICATION_ROOT
from widgets import TimePlotWidget
import pyqtgraph as pg


class UiBuilderMixin:
    """Builds the main window: widget tree, fonts, theme and signal wiring.

    Every panel is created in ``_build_ui`` as a straight-line construction that
    reads in the same order as the visual layout; the colour and spacing values are
    defined in ``config.UI_DEFAULTS`` and applied afterwards by ``_apply_theme``.
    """

    def _build_ui(self) -> None:
        """Construct the entire widget tree.

        Layout is a top-level horizontal splitter:

        * **Left** - a scroll area holding the control tab group (File, Display,
          ST-feature, Audio) and the branding header with ``logo.png``.
        * **Right** - the mode/info toolbar rows, the shared top time-domain panel, and
          the ``analysis_tabs`` widget containing the ``1D Curve`` and ``t-f Plot`` pages.

        The right panel is split into three stacked regions (Plot 1 / Plot 2 / PSD);
        the exact heights are not fixed here but computed later by
        ``LayoutMixin._apply_right_plot_splitter_sizes`` once Qt has laid the widgets
        out, because fixed estimates break at other DPI/resolution settings.

        Note:
            This method is intentionally long: it is a straight-line construction of the
            widget tree and reads top-to-bottom in the same order as the visual layout.
        """

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        main_layout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)

        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)
        self.logo_label = QtWidgets.QLabel()
        self.logo_label.setFixedSize(56, 56)
        self.logo_label.setAlignment(QtCore.Qt.AlignCenter)
        logo_path = APPLICATION_ROOT / "logo.png"
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
        title_font = QtGui.QFont("Times New Roman", 24, QtGui.QFont.Bold)
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
        control_scroll.setMinimumWidth(360)
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
        self.export_format_combo = QtWidgets.QComboBox()
        self.export_format_combo.addItem("NPZ", "npz")
        self.export_format_combo.addItem("TDMS", "tdms")
        self.export_format_combo.addItem("TXT", "txt")
        self.export_directory_edit = QtWidgets.QLineEdit(str(Path.cwd() / "exports"))
        self.export_browse_button = QtWidgets.QPushButton("Export Dir")
        self.export_visible_button = QtWidgets.QPushButton("Export Visible Raw Data")
        self.sample_type_combo = QtWidgets.QComboBox()
        self.sample_type_combo.setEditable(True)
        self.sample_type_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.sample_type_combo.lineEdit().setPlaceholderText("Optional sample type code")
        self.sample_type_combo.setToolTip("Select or input sample type code. Empty means unmarked.")
        self.sample_type_combo.addItem("")
        sample_type_groups = [
            ("断丝 BK", ["BK14", "BK12", "BK40"]),
            ("平稳流噪声 F", ["F0", "F052", "F065", "F130"]),
            ("非平稳流噪声 F+a/b/c", ["F0a", "F052a", "F065a", "F130a"]),
            ("锤击 HM", ["HM12", "HM14"]),
            ("其他 OT", ["OT"]),
        ]
        model = self.sample_type_combo.model()
        for group_name, codes in sample_type_groups:
            heading = f"[{group_name}]"
            self.sample_type_combo.addItem(heading)
            heading_index = self.sample_type_combo.count() - 1
            if hasattr(model, "item"):
                item = model.item(heading_index)
                if item is not None:
                    item.setEnabled(False)
                    item.setToolTip("小类选项: " + ", ".join(codes))
            self._sample_type_major_tooltips[heading_index] = "小类选项: " + ", ".join(codes)
            for code in codes:
                self.sample_type_combo.addItem(code)
        self.sample_type_combo.setCurrentIndex(0)
        self.sample_type_combo.view().entered.connect(self._handle_sample_type_combo_hover)
        directory_layout.addWidget(self.directory_edit, 0, 0, 1, 2)
        directory_layout.addWidget(self.browse_button, 0, 2)
        directory_layout.addWidget(QtWidgets.QLabel("Export Path"), 1, 0)
        directory_layout.addWidget(self.export_directory_edit, 1, 1)
        directory_layout.addWidget(self.export_browse_button, 1, 2)
        directory_layout.addWidget(self.refresh_button, 2, 0)
        directory_layout.addWidget(self.export_format_combo, 2, 1)
        directory_layout.addWidget(self.export_visible_button, 2, 2)
        directory_layout.addWidget(QtWidgets.QLabel("Sample Type"), 3, 0)
        directory_layout.addWidget(self.sample_type_combo, 3, 1, 1, 2)

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
        self.file_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
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
        self.psd_x_min_spin = QtWidgets.QDoubleSpinBox()
        self.psd_x_min_spin.setDecimals(1)
        self.psd_x_min_spin.setRange(0.0, 10_000_000.0)
        self.psd_x_min_spin.setValue(UI_DEFAULTS.display.psd_x_min_hz)
        self.psd_x_max_spin = QtWidgets.QDoubleSpinBox()
        self.psd_x_max_spin.setDecimals(1)
        self.psd_x_max_spin.setRange(0.0, 10_000_000.0)
        self.psd_x_max_spin.setValue(UI_DEFAULTS.display.psd_x_max_hz)
        self.psd_y_min_spin = QtWidgets.QSpinBox()
        self.psd_y_min_spin.setRange(-200, 100)
        self.psd_y_min_spin.setValue(UI_DEFAULTS.display.psd_y_min)
        self.psd_y_max_spin = QtWidgets.QSpinBox()
        self.psd_y_max_spin.setRange(-200, 100)
        self.psd_y_max_spin.setValue(UI_DEFAULTS.display.psd_y_max)
        self.tf_source_combo = QtWidgets.QComboBox()
        self.tf_source_combo.addItem("CH1", TF_SOURCE_CHANNEL_1)
        self.tf_source_combo.addItem("CH2", TF_SOURCE_CHANNEL_2)
        self.tf_source_combo.setMinimumWidth(82)
        self.tf_source_combo.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
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
        self.apply_y_range_button = QtWidgets.QPushButton("Apply Ranges")
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
        filter_layout.addWidget(QtWidgets.QLabel("PSD X Min (Hz)"), 3, 0)
        filter_layout.addWidget(self.psd_x_min_spin, 3, 1)
        filter_layout.addWidget(QtWidgets.QLabel("PSD X Max (Hz)"), 3, 2)
        filter_layout.addWidget(self.psd_x_max_spin, 3, 3)
        filter_layout.addWidget(QtWidgets.QLabel("PSD Y Min"), 4, 0)
        filter_layout.addWidget(self.psd_y_min_spin, 4, 1)
        filter_layout.addWidget(QtWidgets.QLabel("PSD Y Max"), 4, 2)
        filter_layout.addWidget(self.psd_y_max_spin, 4, 3)
        filter_layout.addWidget(QtWidgets.QLabel("t-f Mode"), 5, 0)
        filter_layout.addWidget(self.tf_mode_combo, 5, 1)
        filter_layout.addWidget(QtWidgets.QLabel("t-f Value"), 5, 2)
        filter_layout.addWidget(self.tf_value_scale_combo, 5, 3)
        filter_layout.addWidget(QtWidgets.QLabel("t-f Window (s)"), 6, 0)
        filter_layout.addWidget(self.tf_window_spin, 6, 1)
        filter_layout.addWidget(QtWidgets.QLabel("t-f Overlap (%)"), 6, 2)
        filter_layout.addWidget(self.tf_overlap_spin, 6, 3)
        filter_layout.addWidget(QtWidgets.QLabel("t-f Y Min (Hz)"), 7, 0)
        filter_layout.addWidget(self.tf_y_min_spin, 7, 1)
        filter_layout.addWidget(QtWidgets.QLabel("t-f Y Max (Hz)"), 7, 2)
        filter_layout.addWidget(self.tf_y_max_spin, 7, 3)
        filter_layout.addWidget(QtWidgets.QLabel("t-f Colormap"), 8, 0)
        filter_layout.addWidget(self.tf_colormap_combo, 8, 1)
        filter_layout.addWidget(self.tf_color_auto_checkbox, 8, 2, 1, 2)
        filter_layout.addWidget(QtWidgets.QLabel("t-f Color Min"), 9, 0)
        filter_layout.addWidget(self.tf_color_min_spin, 9, 1)
        filter_layout.addWidget(QtWidgets.QLabel("t-f Color Max"), 9, 2)
        filter_layout.addWidget(self.tf_color_max_spin, 9, 3)
        filter_layout.addWidget(self.apply_filter_button, 10, 0, 1, 4)
        filter_layout.addWidget(self.apply_y_range_button, 11, 0, 1, 4)
        filter_layout.addWidget(self.apply_tf_button, 12, 0, 1, 4)

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
        self.feature_energy_band_low_spin = QtWidgets.QDoubleSpinBox()
        self.feature_energy_band_low_spin.setDecimals(1)
        self.feature_energy_band_low_spin.setRange(0.0, 1e9)
        self.feature_energy_band_low_spin.setValue(UI_DEFAULTS.feature.band_low_hz)
        self.feature_energy_band_high_spin = QtWidgets.QDoubleSpinBox()
        self.feature_energy_band_high_spin.setDecimals(1)
        self.feature_energy_band_high_spin.setRange(0.0, 1e9)
        self.feature_energy_band_high_spin.setValue(UI_DEFAULTS.feature.band_high_hz)
        self.feature_energy2_band_low_spin = QtWidgets.QDoubleSpinBox()
        self.feature_energy2_band_low_spin.setDecimals(1)
        self.feature_energy2_band_low_spin.setRange(0.0, 1e9)
        self.feature_energy2_band_low_spin.setValue(UI_DEFAULTS.feature.band_low_hz)
        self.feature_energy2_band_high_spin = QtWidgets.QDoubleSpinBox()
        self.feature_energy2_band_high_spin.setDecimals(1)
        self.feature_energy2_band_high_spin.setRange(0.0, 1e9)
        self.feature_energy2_band_high_spin.setValue(UI_DEFAULTS.feature.band_high_hz)
        self.feature_energy2_window_spin = QtWidgets.QDoubleSpinBox()
        self.feature_energy2_window_spin.setDecimals(3)
        self.feature_energy2_window_spin.setRange(0.001, 100000.0)
        self.feature_energy2_window_spin.setValue(UI_DEFAULTS.feature.stage1_window_seconds * 1000.0)
        self.feature_energy2_step_spin = QtWidgets.QDoubleSpinBox()
        self.feature_energy2_step_spin.setDecimals(1)
        self.feature_energy2_step_spin.setRange(1.0, 100.0)
        self.feature_energy2_step_spin.setValue(UI_DEFAULTS.feature.step_percent)
        self.feature_energy2_amp_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.feature_energy2_amp_threshold_spin.setDecimals(6)
        self.feature_energy2_amp_threshold_spin.setRange(0.0, 1e12)
        self.feature_energy2_amp_threshold_spin.setValue(UI_DEFAULTS.feature.amplitude_gate)
        self.feature_energy2_sum_window_spin = QtWidgets.QDoubleSpinBox()
        self.feature_energy2_sum_window_spin.setDecimals(3)
        self.feature_energy2_sum_window_spin.setRange(0.001, 100000.0)
        self.feature_energy2_sum_window_spin.setValue(UI_DEFAULTS.feature.stage2_window_seconds * 1000.0)
        self.feature_energy2_sum_step_spin = QtWidgets.QDoubleSpinBox()
        self.feature_energy2_sum_step_spin.setDecimals(1)
        self.feature_energy2_sum_step_spin.setRange(1.0, 100.0)
        self.feature_energy2_sum_step_spin.setValue(UI_DEFAULTS.feature.step_percent)
        self.feature_psd_sum_band_low_spin = QtWidgets.QDoubleSpinBox()
        self.feature_psd_sum_band_low_spin.setDecimals(1)
        self.feature_psd_sum_band_low_spin.setRange(0.0, 1e9)
        self.feature_psd_sum_band_low_spin.setValue(UI_DEFAULTS.feature.psd_sum_band_low_hz)
        self.feature_psd_sum_band_high_spin = QtWidgets.QDoubleSpinBox()
        self.feature_psd_sum_band_high_spin.setDecimals(1)
        self.feature_psd_sum_band_high_spin.setRange(0.0, 1e9)
        self.feature_psd_sum_band_high_spin.setValue(UI_DEFAULTS.feature.psd_sum_band_high_hz)
        self.feature_psd_sum_window_spin = QtWidgets.QDoubleSpinBox()
        self.feature_psd_sum_window_spin.setDecimals(4)
        self.feature_psd_sum_window_spin.setRange(0.0001, 10.0)
        self.feature_psd_sum_window_spin.setValue(UI_DEFAULTS.feature.psd_sum_window_seconds)
        self.feature_psd_sum_step_spin = QtWidgets.QDoubleSpinBox()
        self.feature_psd_sum_step_spin.setDecimals(4)
        self.feature_psd_sum_step_spin.setRange(0.0001, 10.0)
        self.feature_psd_sum_step_spin.setValue(UI_DEFAULTS.feature.psd_sum_step_seconds)
        self.feature_psd_sum_background_spin = QtWidgets.QDoubleSpinBox()
        self.feature_psd_sum_background_spin.setDecimals(3)
        self.feature_psd_sum_background_spin.setRange(0.0001, 1e6)
        self.feature_psd_sum_background_spin.setValue(UI_DEFAULTS.feature.psd_sum_background_seconds)
        self.feature_maxnum_band_low_spin = QtWidgets.QDoubleSpinBox()
        self.feature_maxnum_band_low_spin.setDecimals(1)
        self.feature_maxnum_band_low_spin.setRange(0.0, 1e9)
        self.feature_maxnum_band_low_spin.setValue(UI_DEFAULTS.feature.band_low_hz)
        self.feature_maxnum_band_high_spin = QtWidgets.QDoubleSpinBox()
        self.feature_maxnum_band_high_spin.setDecimals(1)
        self.feature_maxnum_band_high_spin.setRange(0.0, 1e9)
        self.feature_maxnum_band_high_spin.setValue(UI_DEFAULTS.feature.band_high_hz)
        self.feature_maxnum_window_spin = QtWidgets.QDoubleSpinBox()
        self.feature_maxnum_window_spin.setDecimals(3)
        self.feature_maxnum_window_spin.setRange(0.001, 100000.0)
        self.feature_maxnum_window_spin.setValue(UI_DEFAULTS.feature.stage1_window_seconds * 1000.0)
        self.feature_maxnum_step_spin = QtWidgets.QDoubleSpinBox()
        self.feature_maxnum_step_spin.setDecimals(1)
        self.feature_maxnum_step_spin.setRange(1.0, 100.0)
        self.feature_maxnum_step_spin.setValue(UI_DEFAULTS.feature.step_percent)
        self.feature_maxnum_amp_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.feature_maxnum_amp_threshold_spin.setDecimals(6)
        self.feature_maxnum_amp_threshold_spin.setRange(0.0, 1e12)
        self.feature_maxnum_amp_threshold_spin.setValue(UI_DEFAULTS.feature.amplitude_gate)
        self.feature_maxnum_sum_window_spin = QtWidgets.QDoubleSpinBox()
        self.feature_maxnum_sum_window_spin.setDecimals(3)
        self.feature_maxnum_sum_window_spin.setRange(0.001, 100000.0)
        self.feature_maxnum_sum_window_spin.setValue(UI_DEFAULTS.feature.maxnum_window_seconds * 1000.0)
        self.feature_maxnum_sum_step_spin = QtWidgets.QDoubleSpinBox()
        self.feature_maxnum_sum_step_spin.setDecimals(4)
        self.feature_maxnum_sum_step_spin.setRange(0.0001, 10.0)
        self.feature_maxnum_sum_step_spin.setValue(UI_DEFAULTS.feature.maxnum_step_seconds)
        self.feature_maxnum_sub_window_spin = QtWidgets.QDoubleSpinBox()
        self.feature_maxnum_sub_window_spin.setDecimals(3)
        self.feature_maxnum_sub_window_spin.setRange(0.001, 100000.0)
        self.feature_maxnum_sub_window_spin.setValue(UI_DEFAULTS.feature.maxnum_sub_window_seconds * 1000.0)
        self.feature_maxnum_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.feature_maxnum_threshold_spin.setDecimals(2)
        self.feature_maxnum_threshold_spin.setRange(0.1, 1000000.0)
        self.feature_maxnum_threshold_spin.setValue(UI_DEFAULTS.feature.maxnum_threshold)
        self.feature_window_spin = QtWidgets.QDoubleSpinBox()
        self.feature_window_spin.setDecimals(3)
        self.feature_window_spin.setRange(0.001, 100000.0)
        self.feature_window_spin.setValue(UI_DEFAULTS.feature.window_seconds * 1000.0)
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

        ratio_params_widget = QtWidgets.QWidget()
        ratio_params_layout = QtWidgets.QGridLayout(ratio_params_widget)
        ratio_params_layout.setContentsMargins(0, 0, 0, 0)
        ratio_params_layout.setVerticalSpacing(4)
        ratio_params_layout.addWidget(QtWidgets.QLabel("Band 1 Low (Hz)"), 0, 0)
        ratio_params_layout.addWidget(self.feature_num_low_spin, 0, 1)
        ratio_params_layout.addWidget(QtWidgets.QLabel("Band 1 High (Hz)"), 1, 0)
        ratio_params_layout.addWidget(self.feature_num_high_spin, 1, 1)
        ratio_params_layout.addWidget(QtWidgets.QLabel("Band 2 Low (Hz)"), 2, 0)
        ratio_params_layout.addWidget(self.feature_den_low_spin, 2, 1)
        ratio_params_layout.addWidget(QtWidgets.QLabel("Band 2 High (Hz)"), 3, 0)
        ratio_params_layout.addWidget(self.feature_den_high_spin, 3, 1)
        ratio_params_layout.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding), 4, 0)

        energy_params_widget = QtWidgets.QWidget()
        energy_params_layout = QtWidgets.QGridLayout(energy_params_widget)
        energy_params_layout.setContentsMargins(0, 0, 0, 0)
        energy_params_layout.setVerticalSpacing(4)
        energy_params_layout.addWidget(QtWidgets.QLabel("Band Low (Hz)"), 0, 0)
        energy_params_layout.addWidget(self.feature_energy_band_low_spin, 0, 1)
        energy_params_layout.addWidget(QtWidgets.QLabel("Band High (Hz)"), 1, 0)
        energy_params_layout.addWidget(self.feature_energy_band_high_spin, 1, 1)
        energy_params_layout.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding), 2, 0)

        energy2_params_widget = QtWidgets.QWidget()
        energy2_params_layout = QtWidgets.QGridLayout(energy2_params_widget)
        energy2_params_layout.setContentsMargins(0, 0, 0, 0)
        energy2_params_layout.setVerticalSpacing(4)
        energy2_params_layout.addWidget(QtWidgets.QLabel("Band Low (Hz)"), 0, 0)
        energy2_params_layout.addWidget(self.feature_energy2_band_low_spin, 0, 1)
        energy2_params_layout.addWidget(QtWidgets.QLabel("Band High (Hz)"), 1, 0)
        energy2_params_layout.addWidget(self.feature_energy2_band_high_spin, 1, 1)
        energy2_params_layout.addWidget(QtWidgets.QLabel("Stage 1 Window (ms)"), 2, 0)
        energy2_params_layout.addWidget(self.feature_energy2_window_spin, 2, 1)
        energy2_params_layout.addWidget(QtWidgets.QLabel("Stage 1 Step (% of window)"), 3, 0)
        energy2_params_layout.addWidget(self.feature_energy2_step_spin, 3, 1)
        energy2_params_layout.addWidget(QtWidgets.QLabel("Amplitude Gate"), 4, 0)
        energy2_params_layout.addWidget(self.feature_energy2_amp_threshold_spin, 4, 1)
        energy2_params_layout.addWidget(QtWidgets.QLabel("Stage 2 Window (ms)"), 5, 0)
        energy2_params_layout.addWidget(self.feature_energy2_sum_window_spin, 5, 1)
        energy2_params_layout.addWidget(QtWidgets.QLabel("Stage 2 Step (% of window)"), 6, 0)
        energy2_params_layout.addWidget(self.feature_energy2_sum_step_spin, 6, 1)
        energy2_params_layout.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding), 7, 0)

        psd_sum_params_widget = QtWidgets.QWidget()
        psd_sum_params_layout = QtWidgets.QGridLayout(psd_sum_params_widget)
        psd_sum_params_layout.setContentsMargins(0, 0, 0, 0)
        psd_sum_params_layout.setVerticalSpacing(4)
        psd_sum_params_layout.addWidget(QtWidgets.QLabel("Band Low (Hz)"), 0, 0)
        psd_sum_params_layout.addWidget(self.feature_psd_sum_band_low_spin, 0, 1)
        psd_sum_params_layout.addWidget(QtWidgets.QLabel("Band High (Hz)"), 1, 0)
        psd_sum_params_layout.addWidget(self.feature_psd_sum_band_high_spin, 1, 1)
        psd_sum_params_layout.addWidget(QtWidgets.QLabel("PSD Window (s)"), 2, 0)
        psd_sum_params_layout.addWidget(self.feature_psd_sum_window_spin, 2, 1)
        psd_sum_params_layout.addWidget(QtWidgets.QLabel("Step (s)"), 3, 0)
        psd_sum_params_layout.addWidget(self.feature_psd_sum_step_spin, 3, 1)
        psd_sum_params_layout.addWidget(QtWidgets.QLabel("Background Window (s)"), 4, 0)
        psd_sum_params_layout.addWidget(self.feature_psd_sum_background_spin, 4, 1)
        psd_sum_params_layout.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding), 5, 0)

        maxnum_params_widget = QtWidgets.QWidget()
        maxnum_params_layout = QtWidgets.QGridLayout(maxnum_params_widget)
        maxnum_params_layout.setContentsMargins(0, 0, 0, 0)
        maxnum_params_layout.setVerticalSpacing(4)
        maxnum_params_layout.addWidget(QtWidgets.QLabel("Band Low (Hz)"), 0, 0)
        maxnum_params_layout.addWidget(self.feature_maxnum_band_low_spin, 0, 1)
        maxnum_params_layout.addWidget(QtWidgets.QLabel("Band High (Hz)"), 1, 0)
        maxnum_params_layout.addWidget(self.feature_maxnum_band_high_spin, 1, 1)
        maxnum_params_layout.addWidget(QtWidgets.QLabel("Stage 1 Window (ms)"), 2, 0)
        maxnum_params_layout.addWidget(self.feature_maxnum_window_spin, 2, 1)
        maxnum_params_layout.addWidget(QtWidgets.QLabel("Stage 1 Step (% of window)"), 3, 0)
        maxnum_params_layout.addWidget(self.feature_maxnum_step_spin, 3, 1)
        maxnum_params_layout.addWidget(QtWidgets.QLabel("Amplitude Gate"), 4, 0)
        maxnum_params_layout.addWidget(self.feature_maxnum_amp_threshold_spin, 4, 1)
        maxnum_params_layout.addWidget(QtWidgets.QLabel("Stage 2 Window (ms)"), 5, 0)
        maxnum_params_layout.addWidget(self.feature_maxnum_sum_window_spin, 5, 1)
        maxnum_params_layout.addWidget(QtWidgets.QLabel("Stage 2 Step (s)"), 6, 0)
        maxnum_params_layout.addWidget(self.feature_maxnum_sum_step_spin, 6, 1)
        maxnum_params_layout.addWidget(QtWidgets.QLabel("Sub Window (ms)"), 7, 0)
        maxnum_params_layout.addWidget(self.feature_maxnum_sub_window_spin, 7, 1)
        maxnum_params_layout.addWidget(QtWidgets.QLabel("Max Threshold (×1e-6)"), 8, 0)
        maxnum_params_layout.addWidget(self.feature_maxnum_threshold_spin, 8, 1)
        maxnum_params_layout.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding), 9, 0)

        self.feature_params_stack = QtWidgets.QStackedWidget()
        self.feature_params_stack.addWidget(ratio_params_widget)
        self.feature_params_stack.addWidget(energy_params_widget)
        self.feature_params_stack.addWidget(energy2_params_widget)
        self.feature_params_stack.addWidget(psd_sum_params_widget)
        self.feature_params_stack.addWidget(maxnum_params_widget)
        feature_layout.addWidget(self.feature_params_stack, 0, 0, 4, 2)

        self.feature_window_gate_widget = QtWidgets.QWidget()
        window_gate_layout = QtWidgets.QGridLayout(self.feature_window_gate_widget)
        window_gate_layout.setContentsMargins(0, 0, 0, 0)
        window_gate_layout.setVerticalSpacing(4)
        window_gate_layout.addWidget(QtWidgets.QLabel("Window (ms)"), 0, 0)
        window_gate_layout.addWidget(self.feature_window_spin, 0, 1)
        window_gate_layout.addWidget(QtWidgets.QLabel("Step (% of window)"), 1, 0)
        window_gate_layout.addWidget(self.feature_step_spin, 1, 1)
        window_gate_layout.addWidget(QtWidgets.QLabel("Amplitude Gate"), 2, 0)
        window_gate_layout.addWidget(self.feature_amp_threshold_spin, 2, 1)
        feature_layout.addWidget(self.feature_window_gate_widget, 4, 0, 3, 2)
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

        file_tab = QtWidgets.QWidget()
        file_tab_layout = QtWidgets.QVBoxLayout(file_tab)
        file_tab_layout.setContentsMargins(0, 0, 0, 0)
        file_tab_layout.setSpacing(12)
        file_tab_layout.addWidget(directory_group)
        file_tab_layout.addWidget(sort_group, stretch=1)

        control_tabs = QtWidgets.QTabWidget()
        control_tabs.setObjectName("controlTabs")
        control_tabs.addTab(file_tab, "File")
        control_tabs.addTab(filter_group, "Display")
        control_tabs.addTab(feature_group, "ST-feature")
        control_tabs.addTab(audio_group, "Audio")

        control_layout.addWidget(control_tabs, stretch=1)

        right_panel = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        axis = AbsoluteTimeAxis("bottom")
        feature_axis = AbsoluteTimeAxis("bottom")
        self.time_plot = TimePlotWidget(axisItems={"bottom": axis})
        self.feature_plot = TimePlotWidget(axisItems={"bottom": feature_axis})
        time_panel = QtWidgets.QWidget()
        time_layout = QtWidgets.QHBoxLayout(time_panel)
        time_layout.setContentsMargins(0, 0, 0, 0)
        time_layout.setSpacing(0)
        time_layout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        time_column_layout = QtWidgets.QVBoxLayout()
        time_column_layout.setContentsMargins(0, 0, 0, 0)
        time_column_layout.setSpacing(6)
        self._time_column_layout = time_column_layout
        self.zoom_mode_button = QtWidgets.QPushButton("矩形放大")
        self.zoom_mode_button.setCheckable(True)
        self.zoom_mode_button.setChecked(UI_DEFAULTS.view.zoom_mode_checked)
        self.psd_mode_button = QtWidgets.QPushButton("计算PSD")
        self.psd_mode_button.setCheckable(True)
        self.back_view_button = QtWidgets.QPushButton("撤销放大")
        self.back_view_button.setEnabled(False)
        self.fixed_psd_button = QtWidgets.QPushButton("固定PSD窗")
        self.fixed_psd_button.setCheckable(True)
        self.reset_view_button = QtWidgets.QPushButton("重置窗口")
        self.zoom_out_button = QtWidgets.QPushButton("缩小2倍")
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
            button.setMinimumWidth(78)
            button.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.setSpacing(3)
        mode_row.addWidget(self.zoom_mode_button)
        mode_row.addWidget(self.psd_mode_button)
        mode_row.addWidget(self.back_view_button)
        mode_row.addWidget(self.fixed_psd_button)
        mode_row.addWidget(self.zoom_out_button)
        mode_row.addWidget(self.reset_view_button)
        mode_row.addStretch(1)
        self.visible_length_label = QtWidgets.QLabel("Visible: 0.000 s")
        self.window_length_label = QtWidgets.QLabel("Window: 0.000 s")
        self.visible_window_spin = QtWidgets.QDoubleSpinBox()
        self.visible_window_spin.setDecimals(3)
        self.visible_window_spin.setRange(0.001, 1e6)
        self.visible_window_spin.setSingleStep(0.001)
        self.visible_window_spin.setValue(UI_DEFAULTS.view.visible_window_seconds)
        self.apply_visible_window_button = QtWidgets.QPushButton("应用窗宽")
        self.apply_visible_window_button.setMinimumWidth(64)
        self.apply_visible_window_button.setMaximumWidth(96)
        self.feature_plot_mode_combo = QtWidgets.QComboBox()
        self.feature_plot_mode_combo.addItem("None", FEATURE_MODE_NONE)
        self.feature_plot_mode_combo.addItem("CH1", FEATURE_MODE_CHANNEL_1)
        self.feature_plot_mode_combo.addItem("CH2", FEATURE_MODE_CHANNEL_2)
        self.feature_plot_mode_combo.addItem("SVM Prediction", FEATURE_MODE_SVM)
        self.feature_plot_mode_combo.addItem("ST Energy Ratio", FEATURE_MODE_ENERGY)
        self.feature_plot_mode_combo.addItem("ST Energy", FEATURE_MODE_BAND_ENERGY)
        self.feature_plot_mode_combo.addItem("ST Energy Energy", FEATURE_MODE_ENERGY_ENERGY)
        self.feature_plot_mode_combo.addItem("ST PSD sum", FEATURE_MODE_PSD_SUM)
        self.feature_plot_mode_combo.addItem("ST-energy-max-num", FEATURE_MODE_MAX_NUM)
        self.psd_source_combo = QtWidgets.QComboBox()
        self.psd_source_combo.addItem("CH1", PSD_SOURCE_CHANNEL_1)
        self.psd_source_combo.addItem("CH2", PSD_SOURCE_CHANNEL_2)
        self.psd_source_combo.addItem("CH1+CH2", PSD_SOURCE_BOTH)
        self.psd_source_combo.setMinimumWidth(82)
        info_font = QtGui.QFont("Times New Roman", 11)
        self.visible_length_label.setFont(info_font)
        self.window_length_label.setFont(info_font)
        self.time_scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        self.time_scrollbar.setEnabled(False)
        self.time_scrollbar.setSingleStep(1)
        self.time_scrollbar.setPageStep(1)
        info_row = QtWidgets.QHBoxLayout()
        info_row.setSpacing(3)
        info_row.addSpacing(4)
        info_row.addWidget(self.visible_length_label)
        info_row.addSpacing(4)
        info_row.addWidget(self.window_length_label)
        info_row.addSpacing(4)
        info_row.addWidget(QtWidgets.QLabel("窗宽(s)"))
        info_row.addWidget(self.visible_window_spin)
        info_row.addWidget(self.apply_visible_window_button)
        info_row.addSpacing(6)
        info_row.addWidget(QtWidgets.QLabel("PSD"))
        info_row.addWidget(self.psd_source_combo)
        info_row.addSpacing(6)
        info_row.addWidget(QtWidgets.QLabel("CH"))
        info_row.addWidget(self.tf_source_combo)
        info_row.addSpacing(6)
        info_row.addWidget(QtWidgets.QLabel("Plot 2"))
        info_row.addWidget(self.feature_plot_mode_combo)
        info_row.addStretch(1)
        psd_axis = LogPowerFrequencyAxis("bottom")
        self.psd_plot = pg.PlotWidget(axisItems={"bottom": psd_axis})
        tf_axis = AbsoluteTimeAxis("bottom")
        tf_time_axis = AbsoluteTimeAxis("bottom")
        tf_freq_axis = LogFrequencyAxis("left")
        self.tf_time_plot = pg.PlotWidget(axisItems={"bottom": tf_time_axis})
        self.tf_plot = pg.PlotWidget(axisItems={"bottom": tf_axis, "left": tf_freq_axis})
        configure_plot_widget(self.time_plot, "Phase (rad)", "Time")
        configure_plot_widget(self.feature_plot, "SVM Prediction", "Time")
        configure_plot_widget(self.psd_plot, "PSD (dB rad^2/Hz)", "Frequency (Hz)")
        configure_plot_widget(self.tf_time_plot, "Phase (rad)", "Time")
        configure_plot_widget(self.tf_plot, "Frequency (Hz)", "Time")
        self.tf_plot.showGrid(x=True, y=False, alpha=0.22)
        aligned_left_axis_width = 90
        self.time_plot.getPlotItem().getAxis("left").setWidth(aligned_left_axis_width)
        self.feature_plot.getPlotItem().getAxis("left").setWidth(aligned_left_axis_width)
        self.tf_time_plot.getPlotItem().getAxis("left").setWidth(aligned_left_axis_width)
        tf_left_axis = self.tf_plot.getPlotItem().getAxis("left")
        tf_left_axis.setWidth(aligned_left_axis_width)
        tf_left_axis.setStyle(tickLength=8, maxTickLevel=1, maxTextLevel=0, tickAlpha=255, showValues=True)
        self.psd_plot.setLogMode(x=True, y=False)
        self.psd_plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.psd_plot.getViewBox().setMouseEnabled(x=True, y=True)
        self.tf_plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.tf_plot.getViewBox().setMouseEnabled(x=True, y=True)
        self.time_curve = self.time_plot.plot(pen=make_pen("#CC2222", 1))
        self.time_curve.setClipToView(True)
        self.time_curve.setDownsampling(auto=True, method="peak")
        self.time_curve.setSkipFiniteCheck(True)
        self.feature_curve = self.feature_plot.plot(pen=make_pen("#2266AA", 2))
        self.feature_curve.setClipToView(True)
        self.feature_curve.setDownsampling(auto=True, method="peak")
        self.feature_curve.setSkipFiniteCheck(True)
        self.tf_time_curve = self.tf_time_plot.plot(pen=make_pen("#CC2222", 1))
        self.tf_time_curve.setClipToView(True)
        self.tf_time_curve.setDownsampling(auto=True, method="peak")
        self.tf_time_curve.setSkipFiniteCheck(True)
        self.psd_plot.addLegend(offset=(-10, 10))
        self.psd_curve = self.psd_plot.plot(pen=make_pen("#AA3333", 1), name="CH1")
        self.psd_curve.setSkipFiniteCheck(True)
        self.psd_curve_channel_2 = self.psd_plot.plot(pen=make_pen("#2266AA", 1), name="CH2")
        self.psd_curve_channel_2.setSkipFiniteCheck(True)
        self.tf_image_item = pg.ImageItem(axisOrder="row-major")
        self.tf_plot.addItem(self.tf_image_item)
        self.tf_histogram = pg.HistogramLUTWidget()
        self.tf_histogram.setImageItem(self.tf_image_item)
        self.tf_histogram.setBackground("#FFFFFF")
        self.tf_histogram.setStyleSheet("background: #FFFFFF;")
        self.tf_histogram.item.vb.setBackgroundColor("#FFFFFF")
        self.tf_histogram.item.axis.setPen(pg.mkPen("k"))
        self.tf_histogram.item.axis.setTextPen(pg.mkPen("k"))
        self.tf_histogram.item.axis.setStyle(
            tickFont=QtGui.QFont("Times New Roman", AXIS_TICK_FONT_SIZE_PT),
            tickTextOffset=8,
            tickAlpha=255,
        )
        self.tf_histogram.setMinimumWidth(self._tf_side_panel_width)
        self.tf_histogram.setMaximumWidth(self._tf_side_panel_width)
        time_column_layout.addLayout(mode_row)
        time_column_layout.addLayout(info_row)
        time_column_layout.addWidget(self.time_plot, stretch=1)
        time_column_layout.addWidget(self.time_scrollbar)
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
        curve_splitter.setSizes([270, 540])
        self._curve_splitter = curve_splitter
        curve_tab_layout.addWidget(curve_splitter)
        tf_tab = QtWidgets.QWidget()
        tf_tab_layout = QtWidgets.QHBoxLayout(tf_tab)
        tf_tab_layout.setContentsMargins(0, 6, 6, 6)
        tf_tab_layout.setSpacing(6)
        tf_tab_layout.addWidget(self.tf_plot, stretch=1)
        tf_tab_layout.addWidget(self.tf_histogram)
        self.analysis_tabs = QtWidgets.QTabWidget()
        self.analysis_tabs.setObjectName("analysisTabs")
        self.analysis_tabs.setTabPosition(QtWidgets.QTabWidget.South)
        self.analysis_tabs.addTab(curve_tab, "1D Curve")
        self.analysis_tabs.addTab(tf_tab, "t-f Plot")
        right_panel.addWidget(time_panel)
        right_panel.addWidget(self.analysis_tabs)
        self._right_panel_splitter = right_panel
        self._apply_right_plot_splitter_sizes()
        QtCore.QTimer.singleShot(0, self._apply_right_plot_splitter_sizes)
        self._update_time_tf_alignment_for_tab(self.analysis_tabs.currentIndex())
        self.tf_color_min_spin.setEnabled(False)
        self.tf_color_max_spin.setEnabled(False)

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_splitter.addWidget(control_scroll)
        main_splitter.addWidget(right_panel)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        self._main_splitter = main_splitter
        main_layout.addWidget(main_splitter, stretch=1)

        self.statusBar().showMessage("Ready.")
        self._apply_fonts()
        self._apply_theme()
        self._update_interaction_mode()
        self._update_channel_option_controls()
        self._update_feature_plot_style()
        self._update_feature_params_page()
        self._handle_tf_color_auto_toggled(self.tf_color_auto_checkbox.isChecked())
        self._apply_time_frequency_colormap()
        main_splitter = getattr(self, "_main_splitter", None)
        if main_splitter is not None:
            main_splitter.setSizes([self.width() // 6, self.width() - self.width() // 6])


    def _apply_fonts(self) -> None:
        """Apply the shared control font to every widget in the window.
        """

        font = QtGui.QFont("Times New Roman", 10)
        self.setFont(font)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.setFont(font)


    def _apply_initial_window_size(self) -> None:
        """Size the window to the available screen area.

        Accounts for the window frame and title bar, clamps to the usable screen
        geometry, and enforces the minimum size, so the window never opens larger than
        the display (which previously produced a geometry warning on small screens).
        """

        preferred_width = 1500
        preferred_height = 920
        available = QtWidgets.QApplication.primaryScreen().availableGeometry()
        frame_border = 24
        frame_title = 40
        width = min(preferred_width, max(800, available.width() - frame_border))
        height = min(preferred_height, max(600, available.height() - frame_title))
        self.setMinimumSize(800, 600)
        self.resize(width, height)
        main_splitter = getattr(self, "_main_splitter", None)
        if main_splitter is not None:
            main_splitter.setSizes([width // 6, width - width // 6])


    def _apply_theme(self) -> None:
        """Apply the colour theme, spacing and border styling to all widgets.

        Implements the colour rules documented in ``开发需求与规范.txt``: low-saturation
        neutrals for the background and text, a mid-tone blue for primary buttons, and
        3-5 lightness steps per colour for hover/selected/disabled states.
        """

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
            QTabWidget#controlTabs QTabBar::tab {
                min-width: 62px;
                padding: 6px 8px;
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
        """Connect every widget signal to its handler.

        Grouped by panel so the wiring can be read alongside the matching UI block in
        ``_build_ui``. ``_bind_horizontal_scroll_shortcuts`` is called separately
        because it also owns shortcut lifetime bookkeeping.
        """

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
        self.sample_type_combo.currentTextChanged.connect(self._handle_sample_type_text_changed)
        self.sort_field_combo.currentIndexChanged.connect(self._refresh_file_list)
        self.sort_order_combo.currentIndexChanged.connect(self._refresh_file_list)
        self.file_list.itemSelectionChanged.connect(self._handle_file_selection)
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
        self.psd_source_combo.currentIndexChanged.connect(self._handle_psd_source_changed)
        self.tf_source_combo.currentIndexChanged.connect(self._handle_tf_source_changed)
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
        self.time_plot.arrivalMarkRequested.connect(self._mark_arrival_at_index)
        self.feature_plot.windowSelected.connect(self._update_psd_from_selection)
        self.feature_plot.clearSelectionRequested.connect(self._clear_selection)
        self.feature_plot.arrivalMarkRequested.connect(self._mark_arrival_at_index)
        self.time_plot.getViewBox().sigRangeChanged.connect(self._record_view_history)
        self.time_plot.getViewBox().sigRangeChanged.connect(self._update_visible_length_label)
        self.time_plot.getViewBox().sigRangeChanged.connect(self._handle_time_view_changed)
        self.feature_plot.getViewBox().sigRangeChanged.connect(self._handle_time_view_changed)
        self.time_plot.getViewBox().sigXRangeChanged.connect(self._sync_tf_x_from_time)
        self.tf_time_plot.getViewBox().sigXRangeChanged.connect(self._sync_tf_x_from_time)
        self.tf_plot.getViewBox().sigXRangeChanged.connect(self._sync_time_x_from_tf)
        self.time_plot.getViewBox().sigXRangeChanged.connect(self._sync_feature_x_from_time)
        self.feature_plot.getViewBox().sigXRangeChanged.connect(self._sync_time_x_from_feature)
        self.psd_plot.getViewBox().sigXRangeChanged.connect(self._update_psd_x_axis_ticks)
        self.tf_plot.getViewBox().sigYRangeChanged.connect(self._handle_tf_y_range_changed)
        self.time_plot.scene().sigMouseMoved.connect(self._handle_time_plot_mouse_moved)
        self.tf_plot.scene().sigMouseMoved.connect(self._handle_tf_plot_mouse_moved)
        self._bind_horizontal_scroll_shortcuts()


    def _bind_horizontal_scroll_shortcuts(self) -> None:
        """Create and register the left/right pan keyboard shortcuts.

        The created ``QShortcut`` objects are kept on the instance so they are not
        garbage-collected and so they can be removed on close.
        """

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
        for sequence, direction in (
            ("Ctrl+Left", -1),
            ("Ctrl+Right", 1),
        ):
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(sequence), self)
            shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            shortcut.setAutoRepeat(True)
            shortcut.activated.connect(
                lambda direction=direction: self._move_arrival_marker(direction)
            )
            self._scroll_shortcuts.append(shortcut)

