from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from config import UI_DEFAULTS
from data_access import build_export_npz_name, build_export_npz_suffix, list_data_files, load_waveform, save_npz_waveform
from models import FileRecord, FilterMode, LoadedWaveform, SortField
from plotting import (
    AXIS_TICK_FONT_SIZE_PT,
    AbsoluteTimeAxis,
    LogFrequencyAxis,
    configure_plot_widget,
    create_colormap,
    make_pen,
)
from processing import apply_display_filter, compute_time_frequency_map, validate_filter

TF_SCALE_LOG = "log"
TF_SCALE_LINEAR = "linear"


@dataclass(frozen=True)
class SamplePosition:
    file_index: int
    window_index: int


class TimePlotWidget(pg.PlotWidget):
    arrivalMarkRequested = QtCore.pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_length = 0
        self._arrival_line: Optional[pg.InfiniteLine] = None

    def set_data_length(self, size: int) -> None:
        self._data_length = max(0, int(size))

    def set_arrival_marker(self, sample_index: Optional[float]) -> None:
        if sample_index is None:
            if self._arrival_line is not None:
                self.removeItem(self._arrival_line)
                self._arrival_line = None
            return
        bounded = self._bound_x(sample_index)
        if self._arrival_line is None:
            self._arrival_line = pg.InfiniteLine(pos=bounded, angle=90, movable=False, pen=make_pen("#E11D48", 2))
            self.addItem(self._arrival_line)
        else:
            self._arrival_line.setPos(bounded)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.RightButton and bool(event.modifiers() & QtCore.Qt.ControlModifier):
            point = self.plotItem.vb.mapSceneToView(self.mapToScene(event.pos()))
            self.arrivalMarkRequested.emit(self._bound_x(point.x()))
            event.accept()
            return
        super().mousePressEvent(event)

    def _bound_x(self, value: float) -> float:
        if self._data_length <= 1:
            return 0.0
        return min(max(float(value), 0.0), float(self._data_length - 1))


class LabelDialog(QtWidgets.QDialog):
    codeSelected = QtCore.pyqtSignal(str)
    skipRequested = QtCore.pyqtSignal()
    beforeRequested = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("样本标记")
        self.setModal(False)
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
        self.setMinimumWidth(460)
        self._button_layout = QtWidgets.QGridLayout()
        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        self.info_label = QtWidgets.QLabel("请选择当前样本类型。")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        layout.addLayout(self._button_layout)

        actions = QtWidgets.QHBoxLayout()
        actions.setSpacing(12)
        self.before_button = QtWidgets.QPushButton("Before")
        self.skip_button = QtWidgets.QPushButton("Skip")
        self.close_button = QtWidgets.QPushButton("Close")
        self.before_button.setProperty("actionRole", "before")
        self.skip_button.setProperty("actionRole", "skip")
        self.close_button.setProperty("actionRole", "close")
        for button in (self.before_button, self.skip_button, self.close_button):
            button.setMinimumWidth(130)
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        actions.addWidget(self.before_button)
        actions.addWidget(self.skip_button)
        actions.addWidget(self.close_button)
        layout.addLayout(actions)

        self.before_button.clicked.connect(self.beforeRequested.emit)
        self.skip_button.clicked.connect(self.skipRequested.emit)
        self.close_button.clicked.connect(self.hide)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QDialog { background: #F8FAFC; }
            QLabel { color: #0F172A; font-size: 17px; }
            QPushButton {
                background: #DBEAFE;
                color: #1D4ED8;
                border: 1px solid #60A5FA;
                border-radius: 8px;
                padding: 10px 14px;
                min-height: 48px;
                font-size: 17px;
                font-weight: 600;
            }
            QPushButton:hover { background: #BFDBFE; }
            QPushButton:pressed {
                background: #2563EB;
                color: #FFFFFF;
                padding-top: 12px;
                padding-bottom: 8px;
            }
            QPushButton[actionRole="label"] {
                background: #FEF3C7;
                border-color: #F59E0B;
                color: #B45309;
            }
            QPushButton[actionRole="label"]:hover {
                background: #FDE68A;
            }
            QPushButton[actionRole="label"]:pressed {
                background: #D97706;
                color: #FFFFFF;
            }
            """
        )

    def set_codes(self, codes: list[str]) -> None:
        while self._button_layout.count():
            item = self._button_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for index, code in enumerate(codes):
            button = QtWidgets.QPushButton(code)
            button.setProperty("actionRole", "label")
            button.clicked.connect(lambda _checked=False, value=code: self.codeSelected.emit(value))
            self._button_layout.addWidget(button, index // 3, index % 3)

    def set_context_text(self, text: str) -> None:
        self.info_label.setText(text)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("sig_mark")
        self.resize(1500, 920)

        self._source_files: list[FileRecord] = []
        self._current_waveform: Optional[LoadedWaveform] = None
        self._current_filtered_values = np.array([], dtype=np.float64)
        self._current_window_starts = np.array([], dtype=np.int64)
        self._current_position: Optional[SamplePosition] = None
        self._current_window_bounds: tuple[int, int] = (0, 0)
        self._current_arrival_sample_index: Optional[float] = None
        self._last_loaded_file_index: Optional[int] = None
        self._tf_log_freq_bounds: Optional[tuple[float, float]] = None

        self._build_ui()
        self._apply_fonts()
        self._apply_theme()
        self._connect_signals()
        self._label_dialog = LabelDialog(self)
        self._label_dialog.codeSelected.connect(self._label_current_sample)
        self._label_dialog.skipRequested.connect(self._skip_current_sample)
        self._label_dialog.beforeRequested.connect(self._go_to_previous_sample)
        self.statusBar().showMessage("Ready.")
        self._refresh_file_list(auto_select_first=True)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(12)

        header = QtWidgets.QHBoxLayout()
        logo_label = QtWidgets.QLabel()
        logo_path = Path(__file__).resolve().parent.parent / "logo.png"
        if logo_path.exists():
            pixmap = QtGui.QPixmap(str(logo_path)).scaled(56, 56, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
        title = QtWidgets.QLabel("Signal Sample Marker")
        title.setFont(QtGui.QFont("Times New Roman", 20, QtGui.QFont.Bold))
        header_group = QtWidgets.QWidget()
        header_group_layout = QtWidgets.QHBoxLayout(header_group)
        header_group_layout.setContentsMargins(0, 0, 0, 0)
        header_group_layout.setSpacing(10)
        header_group_layout.addWidget(logo_label)
        header_group_layout.addWidget(title)
        header_group.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        header.addWidget(header_group, 0, QtCore.Qt.AlignLeft)
        header.addStretch(1)
        main_layout.addLayout(header)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        control_scroll = QtWidgets.QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(14)
        control_scroll.setWidget(control_panel)
        splitter.addWidget(control_scroll)

        self.directory_group = QtWidgets.QGroupBox("Files")
        directory_layout = QtWidgets.QGridLayout(self.directory_group)
        self.directory_edit = QtWidgets.QLineEdit(str(Path.cwd() / UI_DEFAULTS.file.input_directory))
        self.export_directory_edit = QtWidgets.QLineEdit(str(Path.cwd() / UI_DEFAULTS.file.export_directory))
        self.browse_button = QtWidgets.QPushButton("Browse")
        self.export_browse_button = QtWidgets.QPushButton("Export Dir")
        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.file_list = QtWidgets.QListWidget()
        self.file_list.setMinimumHeight(220)
        self.sort_field_combo = QtWidgets.QComboBox()
        self.sort_field_combo.addItem("Name", SortField.NAME)
        self.sort_field_combo.addItem("Modified Time", SortField.MTIME)
        self.sort_order_combo = QtWidgets.QComboBox()
        self.sort_order_combo.addItem("Ascending", True)
        self.sort_order_combo.addItem("Descending", False)
        directory_layout.addWidget(QtWidgets.QLabel("Input Dir"), 0, 0)
        directory_layout.addWidget(self.directory_edit, 0, 1)
        directory_layout.addWidget(self.browse_button, 0, 2)
        directory_layout.addWidget(QtWidgets.QLabel("Export Dir"), 1, 0)
        directory_layout.addWidget(self.export_directory_edit, 1, 1)
        directory_layout.addWidget(self.export_browse_button, 1, 2)
        directory_layout.addWidget(QtWidgets.QLabel("Sort"), 2, 0)
        directory_layout.addWidget(self.sort_field_combo, 2, 1)
        directory_layout.addWidget(self.sort_order_combo, 2, 2)
        directory_layout.addWidget(self.refresh_button, 3, 0, 1, 3)
        directory_layout.addWidget(self.file_list, 4, 0, 1, 3)

        self.preprocess_group = QtWidgets.QGroupBox("Preprocess")
        preprocess_layout = QtWidgets.QGridLayout(self.preprocess_group)
        self.filter_enabled_checkbox = QtWidgets.QCheckBox("Enable Filter")
        self.filter_enabled_checkbox.setChecked(UI_DEFAULTS.display.filter_enabled)
        self.filter_mode_combo = QtWidgets.QComboBox()
        self.filter_mode_combo.addItem(FilterMode.BANDPASS.value, FilterMode.BANDPASS)
        self.filter_mode_combo.addItem(FilterMode.HIGHPASS.value, FilterMode.HIGHPASS)
        self.filter_mode_combo.addItem(FilterMode.LOWPASS.value, FilterMode.LOWPASS)
        self.filter_mode_combo.setCurrentIndex(UI_DEFAULTS.display.filter_mode_index)
        self.low_cut_spin = QtWidgets.QDoubleSpinBox()
        self.low_cut_spin.setDecimals(1)
        self.low_cut_spin.setRange(0.0, 1e7)
        self.low_cut_spin.setValue(UI_DEFAULTS.display.low_cut_hz)
        self.high_cut_spin = QtWidgets.QDoubleSpinBox()
        self.high_cut_spin.setDecimals(1)
        self.high_cut_spin.setRange(0.0, 1e7)
        self.high_cut_spin.setValue(UI_DEFAULTS.display.high_cut_hz)
        self.apply_filter_button = QtWidgets.QPushButton("Apply Preprocess")
        preprocess_layout.addWidget(self.filter_enabled_checkbox, 0, 0, 1, 4)
        preprocess_layout.addWidget(QtWidgets.QLabel("Filter Mode"), 1, 0)
        preprocess_layout.addWidget(self.filter_mode_combo, 1, 1, 1, 3)
        preprocess_layout.addWidget(QtWidgets.QLabel("Low Cut (Hz)"), 2, 0)
        preprocess_layout.addWidget(self.low_cut_spin, 2, 1)
        preprocess_layout.addWidget(QtWidgets.QLabel("High Cut (Hz)"), 2, 2)
        preprocess_layout.addWidget(self.high_cut_spin, 2, 3)
        preprocess_layout.addWidget(self.apply_filter_button, 3, 0, 1, 4)

        self.sample_group = QtWidgets.QGroupBox("Sample Window")
        sample_layout = QtWidgets.QGridLayout(self.sample_group)
        self.sample_length_spin = QtWidgets.QDoubleSpinBox()
        self.sample_length_spin.setDecimals(4)
        self.sample_length_spin.setRange(0.0001, 10.0)
        self.sample_length_spin.setSingleStep(0.001)
        self.sample_length_spin.setValue(UI_DEFAULTS.sample.sample_length_seconds)
        self.hop_spin = QtWidgets.QDoubleSpinBox()
        self.hop_spin.setDecimals(4)
        self.hop_spin.setRange(0.0001, 10.0)
        self.hop_spin.setSingleStep(0.001)
        self.hop_spin.setValue(UI_DEFAULTS.sample.hop_seconds)
        self.window_jump_spin = QtWidgets.QSpinBox()
        self.window_jump_spin.setMinimum(1)
        self.window_jump_spin.setMaximum(1)
        self.jump_window_button = QtWidgets.QPushButton("Go To Window")
        self.apply_window_button = QtWidgets.QPushButton("Show First Window")
        self.sample_info_label = QtWidgets.QLabel("No sample loaded.")
        self.sample_info_label.setWordWrap(True)
        sample_layout.addWidget(QtWidgets.QLabel("Length (s)"), 0, 0)
        sample_layout.addWidget(self.sample_length_spin, 0, 1)
        sample_layout.addWidget(QtWidgets.QLabel("Hop (s)"), 0, 2)
        sample_layout.addWidget(self.hop_spin, 0, 3)
        sample_layout.addWidget(self.apply_window_button, 1, 0, 1, 2)
        sample_layout.addWidget(self.window_jump_spin, 1, 2)
        sample_layout.addWidget(self.jump_window_button, 1, 3)
        sample_layout.addWidget(self.sample_info_label, 2, 0, 1, 4)

        self.label_group = QtWidgets.QGroupBox("Label Codes")
        label_layout = QtWidgets.QGridLayout(self.label_group)
        self.label_codes_edit = QtWidgets.QPlainTextEdit("\n".join(UI_DEFAULTS.sample.default_codes))
        self.label_codes_edit.setPlaceholderText("One label code per line")
        self.label_codes_edit.setMinimumHeight(84)
        self.label_codes_edit.setMaximumHeight(108)
        self.new_code_edit = QtWidgets.QLineEdit()
        self.new_code_edit.setPlaceholderText("Input")
        self.add_code_button = QtWidgets.QPushButton("Add")
        self.update_buttons_button = QtWidgets.QPushButton("Update Buttons")
        self.open_label_panel_button = QtWidgets.QPushButton("Open Label Panel")
        label_side_layout = QtWidgets.QVBoxLayout()
        label_side_layout.setContentsMargins(0, 0, 0, 0)
        label_side_layout.setSpacing(8)
        label_side_layout.addWidget(self.new_code_edit)
        label_side_layout.addWidget(self.add_code_button)
        label_side_layout.addWidget(self.update_buttons_button)
        label_side_layout.addWidget(self.open_label_panel_button)
        label_side_layout.addStretch(1)
        label_layout.addWidget(self.label_codes_edit, 0, 0)
        label_layout.addLayout(label_side_layout, 0, 1)
        label_layout.setColumnStretch(0, 3)
        label_layout.setColumnStretch(1, 2)

        self.tf_group = QtWidgets.QGroupBox("t-f Display")
        tf_layout = QtWidgets.QGridLayout(self.tf_group)
        self.tf_value_scale_combo = QtWidgets.QComboBox()
        self.tf_value_scale_combo.addItem("Log", TF_SCALE_LOG)
        self.tf_value_scale_combo.addItem("Linear", TF_SCALE_LINEAR)
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
        self.tf_y_min_spin.setRange(0.0, 1e7)
        self.tf_y_min_spin.setValue(UI_DEFAULTS.display.tf_y_min_hz)
        self.tf_y_max_spin = QtWidgets.QDoubleSpinBox()
        self.tf_y_max_spin.setDecimals(1)
        self.tf_y_max_spin.setRange(0.0, 1e7)
        self.tf_y_max_spin.setValue(UI_DEFAULTS.display.tf_y_max_hz)
        self.tf_colormap_combo = QtWidgets.QComboBox()
        self.tf_colormap_combo.setEditable(True)
        self.tf_colormap_combo.addItems(["jet", "hsv", "seismic", "viridis", "plasma", "magma", "inferno", "turbo"])
        self.tf_colormap_combo.setCurrentText(UI_DEFAULTS.display.tf_colormap)
        self.tf_color_auto_checkbox = QtWidgets.QCheckBox("Auto Color Range")
        self.tf_color_auto_checkbox.setChecked(UI_DEFAULTS.display.tf_color_auto)
        self.tf_color_min_spin = QtWidgets.QDoubleSpinBox()
        self.tf_color_min_spin.setDecimals(3)
        self.tf_color_min_spin.setRange(-1e12, 1e12)
        self.tf_color_min_spin.setValue(UI_DEFAULTS.display.tf_color_min)
        self.tf_color_max_spin = QtWidgets.QDoubleSpinBox()
        self.tf_color_max_spin.setDecimals(3)
        self.tf_color_max_spin.setRange(-1e12, 1e12)
        self.tf_color_max_spin.setValue(UI_DEFAULTS.display.tf_color_max)
        self.apply_tf_button = QtWidgets.QPushButton("Apply t-f Params")
        tf_layout.addWidget(QtWidgets.QLabel("Value Scale"), 0, 0)
        tf_layout.addWidget(self.tf_value_scale_combo, 0, 1)
        tf_layout.addWidget(QtWidgets.QLabel("Window (s)"), 0, 2)
        tf_layout.addWidget(self.tf_window_spin, 0, 3)
        tf_layout.addWidget(QtWidgets.QLabel("Overlap (%)"), 1, 0)
        tf_layout.addWidget(self.tf_overlap_spin, 1, 1)
        tf_layout.addWidget(QtWidgets.QLabel("Colormap"), 1, 2)
        tf_layout.addWidget(self.tf_colormap_combo, 1, 3)
        tf_layout.addWidget(QtWidgets.QLabel("Y Min (Hz)"), 2, 0)
        tf_layout.addWidget(self.tf_y_min_spin, 2, 1)
        tf_layout.addWidget(QtWidgets.QLabel("Y Max (Hz)"), 2, 2)
        tf_layout.addWidget(self.tf_y_max_spin, 2, 3)
        tf_layout.addWidget(QtWidgets.QLabel("Color Min"), 3, 0)
        tf_layout.addWidget(self.tf_color_min_spin, 3, 1)
        tf_layout.addWidget(QtWidgets.QLabel("Color Max"), 3, 2)
        tf_layout.addWidget(self.tf_color_max_spin, 3, 3)
        tf_layout.addWidget(self.tf_color_auto_checkbox, 4, 0, 1, 4)
        tf_layout.addWidget(self.apply_tf_button, 5, 0, 1, 4)

        control_layout.addWidget(self.directory_group)
        control_layout.addWidget(self.preprocess_group)
        control_layout.addWidget(self.sample_group)
        control_layout.addWidget(self.label_group)
        control_layout.addWidget(self.tf_group)
        control_layout.addStretch(1)

        right_panel = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(right_panel)
        splitter.setSizes([440, 1060])

        time_axis = AbsoluteTimeAxis("bottom")
        self.time_plot = TimePlotWidget(axisItems={"bottom": time_axis})
        configure_plot_widget(self.time_plot, "Phase (rad)", "Time")
        self.time_curve = self.time_plot.plot(pen=make_pen("#C81E1E", 1))
        self.time_curve.setClipToView(True)
        self.time_curve.setDownsampling(auto=True, method="peak")
        self.time_curve.setSkipFiniteCheck(True)

        tf_axis = AbsoluteTimeAxis("bottom")
        tf_freq_axis = LogFrequencyAxis("left")
        self.tf_plot = pg.PlotWidget(axisItems={"bottom": tf_axis, "left": tf_freq_axis})
        configure_plot_widget(self.tf_plot, "Frequency (Hz)", "Time")
        self.tf_plot.showGrid(x=True, y=False, alpha=0.22)
        aligned_left_axis_width = 90
        self.time_plot.getPlotItem().getAxis("left").setWidth(aligned_left_axis_width)
        self.tf_plot.getPlotItem().getAxis("left").setWidth(aligned_left_axis_width)
        self.tf_image_item = pg.ImageItem(axisOrder="row-major")
        self.tf_plot.addItem(self.tf_image_item)
        self.tf_histogram = pg.HistogramLUTWidget()
        self.tf_histogram.setImageItem(self.tf_image_item)
        self.tf_histogram.setMinimumWidth(110)
        self.tf_histogram.setMaximumWidth(110)
        self.tf_histogram.setBackground("#FFFFFF")
        self.tf_histogram.item.axis.setPen(pg.mkPen("k"))
        self.tf_histogram.item.axis.setTextPen(pg.mkPen("k"))
        self.tf_histogram.item.axis.setStyle(
            tickFont=QtGui.QFont("Times New Roman", AXIS_TICK_FONT_SIZE_PT),
            tickTextOffset=8,
            tickAlpha=255,
        )
        self.time_right_spacer = QtWidgets.QWidget()
        self.time_right_spacer.setMinimumWidth(110)
        self.time_right_spacer.setMaximumWidth(110)
        self.time_right_spacer.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)

        time_panel = QtWidgets.QWidget()
        time_layout = QtWidgets.QHBoxLayout(time_panel)
        time_layout.setContentsMargins(0, 0, 0, 0)
        time_layout.setSpacing(6)
        time_left = QtWidgets.QWidget()
        time_left_layout = QtWidgets.QVBoxLayout(time_left)
        time_left_layout.setContentsMargins(0, 0, 0, 0)
        time_left_layout.setSpacing(6)
        time_tip = QtWidgets.QLabel("Ctrl + Right Click: mark first arrival time in current sample")
        time_tip.setStyleSheet("color: #475569;")
        time_left_layout.addWidget(time_tip)
        time_left_layout.addWidget(self.time_plot, stretch=1)
        time_layout.addWidget(time_left, stretch=1)
        time_layout.addWidget(self.time_right_spacer)

        tf_panel = QtWidgets.QWidget()
        tf_layout_outer = QtWidgets.QHBoxLayout(tf_panel)
        tf_layout_outer.setContentsMargins(0, 0, 0, 0)
        tf_layout_outer.setSpacing(6)
        tf_layout_outer.addWidget(self.tf_plot, stretch=1)
        tf_layout_outer.addWidget(self.tf_histogram)

        right_panel.addWidget(time_panel)
        right_panel.addWidget(tf_panel)
        right_panel.setSizes([410, 430])

        self.tf_color_min_spin.setEnabled(False)
        self.tf_color_max_spin.setEnabled(False)
        self._apply_time_frequency_colormap()

    def _apply_fonts(self) -> None:
        self.setFont(QtGui.QFont("SimSun", 10))

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #F5F7FA; color: #1F2937; }
            QStatusBar { background: #EEF2F7; color: #475569; border-top: 1px solid #D7DEE8; }
            QGroupBox {
                background: #FFFFFF;
                border: 1px solid #94A3B8;
                border-radius: 8px;
                margin-top: 10px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                background: #FFFFFF;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QPlainTextEdit {
                background: #FFFFFF;
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                padding: 6px 8px;
            }
            QPushButton {
                background: #DCE7F3;
                border: 1px solid #94A3B8;
                border-radius: 8px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover { background: #C7D7EA; }
            QPushButton:pressed { background: #31557A; color: #FFFFFF; }
            """
        )

    def _connect_signals(self) -> None:
        self.browse_button.clicked.connect(self._choose_input_directory)
        self.export_browse_button.clicked.connect(self._choose_export_directory)
        self.refresh_button.clicked.connect(lambda: self._refresh_file_list(auto_select_first=True))
        self.file_list.currentRowChanged.connect(self._handle_file_selection)
        self.apply_filter_button.clicked.connect(self._apply_preprocess_and_reset)
        self.apply_window_button.clicked.connect(self._reset_to_first_window)
        self.jump_window_button.clicked.connect(self._jump_to_window_index)
        self.add_code_button.clicked.connect(self._add_code_from_input)
        self.update_buttons_button.clicked.connect(self._refresh_label_dialog_buttons)
        self.open_label_panel_button.clicked.connect(self._show_label_dialog)
        self.apply_tf_button.clicked.connect(self._refresh_current_sample_view)
        self.tf_color_auto_checkbox.toggled.connect(self._handle_tf_color_auto_toggled)
        self.tf_colormap_combo.currentTextChanged.connect(lambda _text: self._apply_time_frequency_colormap())
        self.time_plot.arrivalMarkRequested.connect(self._mark_arrival_at_index)
        self.tf_plot.getViewBox().sigYRangeChanged.connect(self._handle_tf_y_range_changed)

    def _choose_input_directory(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Input Directory", self.directory_edit.text().strip())
        if directory:
            self.directory_edit.setText(directory)
            self._refresh_file_list(auto_select_first=True)

    def _choose_export_directory(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Export Directory", self.export_directory_edit.text().strip())
        if directory:
            self.export_directory_edit.setText(directory)

    def _refresh_file_list(self, *, auto_select_first: bool) -> None:
        directory = Path(self.directory_edit.text().strip())
        try:
            sort_field = self.sort_field_combo.currentData() or SortField.NAME
            ascending = bool(self.sort_order_combo.currentData())
            self._source_files = list_data_files(directory, sort_field=sort_field, ascending=ascending)
        except Exception as exc:
            self._source_files = []
            self.file_list.clear()
            self.statusBar().showMessage(f"Failed to list files: {exc}")
            return

        self.file_list.blockSignals(True)
        self.file_list.clear()
        for record in self._source_files:
            self.file_list.addItem(record.name)
        self.file_list.blockSignals(False)

        if not self._source_files:
            self._current_waveform = None
            self._current_filtered_values = np.array([], dtype=np.float64)
            self._current_window_starts = np.array([], dtype=np.int64)
            self._current_position = None
            self._clear_plots()
            self.sample_info_label.setText("No data files found.")
            self.statusBar().showMessage("No readable files found in input directory.")
            return

        if auto_select_first:
            self.file_list.setCurrentRow(0)
        elif self._last_loaded_file_index is not None and 0 <= self._last_loaded_file_index < len(self._source_files):
            self.file_list.setCurrentRow(self._last_loaded_file_index)

    def _handle_file_selection(self, row: int) -> None:
        if row < 0 or row >= len(self._source_files):
            return
        self._load_file_by_index(row, reset_window=True)

    def _load_file_by_index(self, file_index: int, *, reset_window: bool) -> bool:
        if file_index < 0 or file_index >= len(self._source_files):
            return False
        record = self._source_files[file_index]
        try:
            waveform = load_waveform(record.path)
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to load file: {exc}")
            return False

        self._current_waveform = waveform
        self._last_loaded_file_index = file_index
        self._current_arrival_sample_index = None
        self._apply_preprocess(reset_window=reset_window)
        return True

    def _apply_preprocess_and_reset(self) -> None:
        self._apply_preprocess(reset_window=True)

    def _apply_preprocess(self, *, reset_window: bool) -> None:
        if self._current_waveform is None:
            self.statusBar().showMessage("Load a waveform first.")
            return

        mode = self.filter_mode_combo.currentData() or FilterMode.HIGHPASS
        low_cut_hz = float(self.low_cut_spin.value())
        high_cut_hz = float(self.high_cut_spin.value())
        valid, message = validate_filter(
            self.filter_enabled_checkbox.isChecked(),
            mode,
            self._current_waveform.sample_rate,
            low_cut_hz,
            high_cut_hz,
        )
        if not valid:
            self.statusBar().showMessage(message or "Invalid filter parameters.")
            return

        self._current_filtered_values = apply_display_filter(
            self._current_waveform.phase_data,
            self._current_waveform.sample_rate,
            self.filter_enabled_checkbox.isChecked(),
            mode,
            low_cut_hz,
            high_cut_hz,
        )
        self._rebuild_window_positions()
        if reset_window:
            self._current_position = SamplePosition(self._last_loaded_file_index or 0, 0)
            self._show_current_position()

    def _rebuild_window_positions(self) -> None:
        if self._current_waveform is None:
            self._current_window_starts = np.array([], dtype=np.int64)
            self.window_jump_spin.setMaximum(1)
            self.window_jump_spin.setValue(1)
            return
        sample_count = int(self._current_filtered_values.size)
        sample_len = max(1, int(round(self.sample_length_spin.value() * self._current_waveform.sample_rate)))
        hop_len = max(1, int(round(self.hop_spin.value() * self._current_waveform.sample_rate)))
        if sample_count <= sample_len:
            self._current_window_starts = np.array([0], dtype=np.int64)
            self.window_jump_spin.setMaximum(1)
            self.window_jump_spin.setValue(1)
            return
        starts = np.arange(0, sample_count - sample_len + 1, hop_len, dtype=np.int64)
        if starts.size == 0 or int(starts[-1]) != sample_count - sample_len:
            starts = np.append(starts, sample_count - sample_len)
        self._current_window_starts = starts.astype(np.int64, copy=False)
        self.window_jump_spin.setMaximum(max(1, int(self._current_window_starts.size)))
        if self._current_position is not None:
            self.window_jump_spin.setValue(min(self.window_jump_spin.maximum(), self._current_position.window_index + 1))
        else:
            self.window_jump_spin.setValue(1)

    def _reset_to_first_window(self) -> None:
        if self._current_waveform is None:
            self.statusBar().showMessage("Load a waveform first.")
            return
        self._rebuild_window_positions()
        self._current_position = SamplePosition(self._last_loaded_file_index or 0, 0)
        self._show_current_position()

    def _jump_to_window_index(self) -> None:
        if self._current_waveform is None or self._last_loaded_file_index is None:
            self.statusBar().showMessage("Load a waveform first.")
            return
        if self._current_window_starts.size == 0:
            self.statusBar().showMessage("No available windows.")
            return
        target_index = int(self.window_jump_spin.value()) - 1
        target_index = min(max(target_index, 0), int(self._current_window_starts.size) - 1)
        self._current_position = SamplePosition(self._last_loaded_file_index, target_index)
        self._show_current_position()

    def _show_current_position(self) -> None:
        if self._current_position is None:
            return
        if self._current_position.file_index != self._last_loaded_file_index:
            if not self._load_file_by_index(self._current_position.file_index, reset_window=False):
                return
        if self._current_waveform is None or self._current_window_starts.size == 0:
            self._clear_plots()
            return

        bounded_window = min(max(self._current_position.window_index, 0), int(self._current_window_starts.size) - 1)
        self._current_position = SamplePosition(self._current_position.file_index, bounded_window)
        self.window_jump_spin.setMaximum(max(1, int(self._current_window_starts.size)))
        self.window_jump_spin.setValue(self._current_position.window_index + 1)
        self._current_arrival_sample_index = None
        self._refresh_current_sample_view()
        self._refresh_label_dialog_buttons()
        self._show_label_dialog()

    def _refresh_current_sample_view(self) -> None:
        if self._current_waveform is None or self._current_position is None or self._current_window_starts.size == 0:
            self._clear_plots()
            return

        sample_rate = float(self._current_waveform.sample_rate)
        sample_len = max(1, int(round(self.sample_length_spin.value() * sample_rate)))
        start_index = int(self._current_window_starts[self._current_position.window_index])
        end_index = min(int(self._current_filtered_values.size), start_index + sample_len)
        segment = np.asarray(self._current_filtered_values[start_index:end_index], dtype=np.float64)
        if segment.size == 0:
            self._clear_plots()
            return

        self._current_window_bounds = (start_index, end_index)
        self.time_plot.set_data_length(segment.size)
        self.time_plot.set_arrival_marker(self._current_arrival_sample_index)
        self.time_curve.setData(np.arange(segment.size, dtype=np.float64), segment)

        time_axis = self.time_plot.getPlotItem().getAxis("bottom")
        if isinstance(time_axis, AbsoluteTimeAxis):
            segment_start = self._current_waveform.start_time + timedelta(seconds=start_index / sample_rate)
            time_axis.set_context(segment_start, sample_rate)
        self.time_plot.setXRange(0.0, max(float(segment.size - 1), 1.0), padding=0.0)
        self.time_plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

        self._update_time_frequency_plot(segment, start_index)
        self._sync_file_selection()
        self._update_sample_info()

    def _build_time_frequency_display_grid(
        self,
        freqs: np.ndarray,
        values_map: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        positive_mask = np.asarray(freqs, dtype=np.float64) > 0.0
        if not np.any(positive_mask):
            return np.array([], dtype=np.float64), np.empty((0, 0), dtype=np.float64)

        positive_freqs = np.asarray(freqs[positive_mask], dtype=np.float64)
        positive_values = np.asarray(values_map[positive_mask, :], dtype=np.float64)
        if positive_freqs.size < 2 or positive_values.size == 0:
            return np.array([], dtype=np.float64), np.empty((0, 0), dtype=np.float64)

        log_freqs = np.log10(positive_freqs)
        display_log_freqs = np.linspace(log_freqs[0], log_freqs[-1], positive_freqs.size, dtype=np.float64)
        display_values = np.empty((display_log_freqs.size, positive_values.shape[1]), dtype=np.float64)
        for column_index in range(positive_values.shape[1]):
            display_values[:, column_index] = np.interp(
                display_log_freqs,
                log_freqs,
                positive_values[:, column_index],
            )
        return display_log_freqs, display_values

    def _update_time_frequency_axis_ticks(self) -> None:
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
            for factor in range(2, 10):
                sub_tick = decade + float(np.log10(factor))
                if y_min <= sub_tick <= y_max:
                    minor_ticks.append((sub_tick, ""))

        axis.setTicks([major_ticks, minor_ticks])

    def _handle_tf_y_range_changed(self, *_args) -> None:
        if self._tf_log_freq_bounds is None:
            return
        self._update_time_frequency_axis_ticks()

    def _update_time_frequency_plot(self, segment: np.ndarray, absolute_start_index: int) -> None:
        if self._current_waveform is None:
            self.tf_image_item.setImage(np.empty((0, 0)))
            self._tf_log_freq_bounds = None
            self.tf_plot.getPlotItem().getAxis("left").setTicks(None)
            return

        sample_rate = float(self._current_waveform.sample_rate)
        freqs, centers, values_map = compute_time_frequency_map(
            segment,
            sample_rate,
            window_seconds=float(self.tf_window_spin.value()),
            overlap_ratio=float(self.tf_overlap_spin.value()) / 100.0,
            spectrum_mode="psd",
        )
        if freqs.size == 0 or centers.size == 0 or values_map.size == 0:
            self.tf_image_item.setImage(np.empty((0, 0)))
            self._tf_log_freq_bounds = None
            self.tf_plot.getPlotItem().getAxis("left").setTicks(None)
            self.statusBar().showMessage("Current sample window is too short for t-f computation.")
            return

        display_log_freqs, display_values_map = self._build_time_frequency_display_grid(freqs, values_map)
        if display_log_freqs.size == 0 or display_values_map.size == 0:
            self.tf_image_item.setImage(np.empty((0, 0)))
            self._tf_log_freq_bounds = None
            self.tf_plot.getPlotItem().getAxis("left").setTicks(None)
            self.statusBar().showMessage("No positive frequency bins available for t-f display.")
            return

        plot_values = display_values_map
        if str(self.tf_value_scale_combo.currentData() or TF_SCALE_LOG) == TF_SCALE_LOG:
            floor = np.finfo(np.float64).tiny
            plot_values = 10.0 * np.log10(np.maximum(plot_values, floor))

        self.tf_image_item.setImage(plot_values, autoLevels=False)
        y0 = float(display_log_freqs[0])
        y1 = float(display_log_freqs[-1])
        self._tf_log_freq_bounds = (y0, y1)
        left_axis = self.tf_plot.getPlotItem().getAxis("left")
        if isinstance(left_axis, LogFrequencyAxis):
            left_axis.setRange(y0, y1)
        x0 = 0.0
        x_span = max(float(segment.size - 1), 1.0)
        y_span = y1 - y0 if y1 != y0 else 1.0
        self.tf_image_item.resetTransform()
        self.tf_image_item.setRect(QtCore.QRectF(x0, y0, x_span, y_span))

        tf_axis = self.tf_plot.getPlotItem().getAxis("bottom")
        if isinstance(tf_axis, AbsoluteTimeAxis):
            segment_start = self._current_waveform.start_time + timedelta(seconds=absolute_start_index / sample_rate)
            tf_axis.set_context(segment_start, sample_rate)

        self.tf_plot.setXRange(x0, x0 + x_span, padding=0.0)

        y_min = float(self.tf_y_min_spin.value())
        y_max = float(self.tf_y_max_spin.value())
        if y_max > y_min > 0:
            self.tf_plot.setYRange(np.log10(y_min), np.log10(y_max), padding=0.0)
        else:
            self.tf_plot.setYRange(y0, y1, padding=0.0)
        self._update_time_frequency_axis_ticks()

        self._apply_time_frequency_colormap()
        if self.tf_color_auto_checkbox.isChecked():
            self.tf_image_item.setLevels((float(np.nanmin(plot_values)), float(np.nanmax(plot_values))))
        else:
            self.tf_image_item.setLevels((float(self.tf_color_min_spin.value()), float(self.tf_color_max_spin.value())))

    def _apply_time_frequency_colormap(self) -> None:
        cmap = create_colormap(self.tf_colormap_combo.currentText())
        self.tf_image_item.setColorMap(cmap)
        # `getGradient()` returns a QLinearGradient in newer pyqtgraph versions,
        # while HistogramLUTWidget.restoreState expects the editor state dict.
        self.tf_histogram.gradient.setColorMap(cmap)

    def _handle_tf_color_auto_toggled(self, checked: bool) -> None:
        self.tf_color_min_spin.setEnabled(not checked)
        self.tf_color_max_spin.setEnabled(not checked)
        self._refresh_current_sample_view()

    def _mark_arrival_at_index(self, sample_index: float) -> None:
        self._current_arrival_sample_index = float(sample_index)
        self.time_plot.set_arrival_marker(self._current_arrival_sample_index)
        if self._current_waveform is None:
            return
        start_index, _ = self._current_window_bounds
        absolute_index = start_index + self._current_arrival_sample_index
        arrival_time = self._current_waveform.start_time + timedelta(seconds=absolute_index / self._current_waveform.sample_rate)
        self.statusBar().showMessage(f"Arrival marked at {arrival_time.strftime('%Y%m%d%H%M%S.%f')[:-2]}")

    def _update_sample_info(self) -> None:
        if self._current_waveform is None or self._current_position is None:
            self.sample_info_label.setText("No sample loaded.")
            return
        start_index, end_index = self._current_window_bounds
        sample_rate = float(self._current_waveform.sample_rate)
        segment_start = self._current_waveform.start_time + timedelta(seconds=start_index / sample_rate)
        segment_end = self._current_waveform.start_time + timedelta(seconds=end_index / sample_rate)
        total_windows = int(self._current_window_starts.size)
        self.sample_info_label.setText(
            f"File {self._current_position.file_index + 1}/{len(self._source_files)} | "
            f"Window {self._current_position.window_index + 1}/{total_windows}\n"
            f"{self._current_waveform.path.name}\n"
            f"{segment_start.strftime('%H:%M:%S.%f')[:-3]} -> {segment_end.strftime('%H:%M:%S.%f')[:-3]}"
        )
        self._label_dialog.set_context_text(
            f"{self._current_waveform.path.name}\n"
            f"Window {self._current_position.window_index + 1}/{total_windows}. "
            "Select a type to save this sample and move to the next one."
        )

    def _show_label_dialog(self) -> None:
        if self._current_position is None:
            return
        self._label_dialog.show()
        self._label_dialog.raise_()
        self._label_dialog.activateWindow()

    def _refresh_label_dialog_buttons(self) -> None:
        self._label_dialog.set_codes(self._collect_label_codes())

    def _collect_label_codes(self) -> list[str]:
        lines = [line.strip().upper() for line in self.label_codes_edit.toPlainText().splitlines()]
        codes: list[str] = []
        for line in lines:
            if line and line not in codes:
                codes.append(line)
        if not codes:
            codes = list(UI_DEFAULTS.sample.default_codes)
            self.label_codes_edit.setPlainText("\n".join(codes))
        return codes

    def _add_code_from_input(self) -> None:
        code = self.new_code_edit.text().strip().upper()
        if not code:
            return
        codes = self._collect_label_codes()
        if code not in codes:
            codes.append(code)
            self.label_codes_edit.setPlainText("\n".join(codes))
            self._refresh_label_dialog_buttons()
        self.new_code_edit.clear()

    def _label_current_sample(self, sample_type: str) -> None:
        if self._current_waveform is None or self._current_position is None:
            return
        export_directory = self.export_directory_edit.text().strip()
        if not export_directory:
            self.statusBar().showMessage("Please set an export directory first.")
            return

        start_index, end_index = self._current_window_bounds
        segment = np.asarray(self._current_filtered_values[start_index:end_index], dtype=np.float64)
        segment_start_time = self._current_waveform.start_time + timedelta(seconds=start_index / self._current_waveform.sample_rate)
        arrival_time = None
        if self._current_arrival_sample_index is not None:
            absolute_index = start_index + self._current_arrival_sample_index
            arrival_time = self._current_waveform.start_time + timedelta(seconds=absolute_index / self._current_waveform.sample_rate)

        export_directory_path = Path(export_directory)
        file_suffix = build_export_npz_suffix(segment_start_time, self._current_waveform.sample_rate)
        for existing_path in export_directory_path.glob(f"*{file_suffix}"):
            try:
                existing_path.unlink()
            except Exception as exc:
                self.statusBar().showMessage(f"Failed to replace previous label: {exc}")
                return

        filename = build_export_npz_name(
            segment_start_time,
            self._current_waveform.sample_rate,
            sample_type=sample_type,
        )
        destination = export_directory_path / filename
        try:
            save_npz_waveform(
                destination,
                segment,
                self._current_waveform.sample_rate,
                segment_start_time,
                arrival_time=arrival_time,
                sample_type=sample_type,
            )
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to save sample: {exc}")
            return

        self.statusBar().showMessage(f"Saved {sample_type} -> {destination.name}")
        self._advance_to_next_sample()

    def _skip_current_sample(self) -> None:
        self.statusBar().showMessage("Current sample skipped.")
        self._advance_to_next_sample()

    def _advance_to_next_sample(self) -> None:
        if self._current_position is None:
            return
        next_position = self._find_next_position(self._current_position)
        if next_position is None:
            self.statusBar().showMessage("All samples finished.")
            self._label_dialog.hide()
            return
        self._current_position = next_position
        self._show_current_position()

    def _go_to_previous_sample(self) -> None:
        if self._current_position is None:
            return
        previous_position = self._find_previous_position(self._current_position)
        if previous_position is None:
            self.statusBar().showMessage("Already at the first sample.")
            return
        self._current_position = previous_position
        self._show_current_position()

    def _find_next_position(self, position: SamplePosition) -> Optional[SamplePosition]:
        if position.file_index == self._last_loaded_file_index and position.window_index + 1 < int(self._current_window_starts.size):
            return SamplePosition(position.file_index, position.window_index + 1)
        for file_index in range(position.file_index + 1, len(self._source_files)):
            if self._load_file_by_index(file_index, reset_window=False) and self._current_window_starts.size > 0:
                return SamplePosition(file_index, 0)
        return None

    def _find_previous_position(self, position: SamplePosition) -> Optional[SamplePosition]:
        if position.file_index == self._last_loaded_file_index and position.window_index > 0:
            return SamplePosition(position.file_index, position.window_index - 1)
        for file_index in range(position.file_index - 1, -1, -1):
            if self._load_file_by_index(file_index, reset_window=False) and self._current_window_starts.size > 0:
                return SamplePosition(file_index, int(self._current_window_starts.size) - 1)
        return None

    def _sync_file_selection(self) -> None:
        if self._last_loaded_file_index is None:
            return
        self.file_list.blockSignals(True)
        self.file_list.setCurrentRow(self._last_loaded_file_index)
        self.file_list.blockSignals(False)

    def _clear_plots(self) -> None:
        self.time_curve.setData([], [])
        self.time_plot.set_arrival_marker(None)
        self.tf_image_item.setImage(np.empty((0, 0)))
        self._tf_log_freq_bounds = None
        self.tf_plot.getPlotItem().getAxis("left").setTicks(None)
