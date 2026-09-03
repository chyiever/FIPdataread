from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from PyQt5 import QtGui
import pyqtgraph as pg

AXIS_LABEL_FONT_SIZE_PT = 13
AXIS_TICK_FONT_SIZE_PT = 13
TIME_AXIS_HEIGHT_PX = 62


_FALLBACK_COLORMAPS: dict[str, list[tuple[float, tuple[int, int, int]]]] = {
    "jet": [
        (0.00, (0, 0, 128)),
        (0.12, (0, 0, 255)),
        (0.35, (0, 255, 255)),
        (0.50, (255, 255, 0)),
        (0.75, (255, 0, 0)),
        (1.00, (128, 0, 0)),
    ],
    "hsv": [
        (0.00, (255, 0, 0)),
        (0.17, (255, 255, 0)),
        (0.33, (0, 255, 0)),
        (0.50, (0, 255, 255)),
        (0.67, (0, 0, 255)),
        (0.83, (255, 0, 255)),
        (1.00, (255, 0, 0)),
    ],
    "seismic": [
        (0.00, (0, 0, 76)),
        (0.25, (0, 0, 255)),
        (0.50, (255, 255, 255)),
        (0.75, (255, 0, 0)),
        (1.00, (76, 0, 0)),
    ],
    "viridis": [
        (0.00, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.50, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.00, (253, 231, 37)),
    ],
    "plasma": [
        (0.00, (13, 8, 135)),
        (0.25, (126, 3, 168)),
        (0.50, (204, 71, 120)),
        (0.75, (248, 149, 64)),
        (1.00, (240, 249, 33)),
    ],
    "magma": [
        (0.00, (0, 0, 4)),
        (0.25, (80, 18, 123)),
        (0.50, (182, 54, 121)),
        (0.75, (251, 136, 97)),
        (1.00, (252, 253, 191)),
    ],
    "inferno": [
        (0.00, (0, 0, 4)),
        (0.25, (87, 15, 109)),
        (0.50, (187, 55, 84)),
        (0.75, (249, 142, 8)),
        (1.00, (252, 255, 164)),
    ],
    "turbo": [
        (0.00, (48, 18, 59)),
        (0.15, (50, 101, 220)),
        (0.35, (26, 199, 194)),
        (0.55, (141, 235, 70)),
        (0.75, (254, 194, 37)),
        (1.00, (122, 4, 3)),
    ],
    "gray": [
        (0.00, (0, 0, 0)),
        (1.00, (255, 255, 255)),
    ],
}


def _append_axis_tick_stubs(axis_item: pg.AxisItem, tick_specs, bounds, tick_levels) -> None:
    dif = axis_item.range[1] - axis_item.range[0]
    if dif == 0:
        return

    orientation = axis_item.orientation
    if orientation == "left":
        scale = -bounds.height() / dif
        offset = axis_item.range[0] * scale - bounds.height()
        axis_index = 0
        tick_stop = bounds.right()
        tick_dir = -1
        visible_min, visible_max = sorted([x * scale - offset for x in axis_item.range])
    elif orientation == "bottom":
        scale = bounds.width() / dif
        offset = axis_item.range[0] * scale
        axis_index = 1
        tick_stop = bounds.top()
        tick_dir = 1
        visible_min, visible_max = sorted([x * scale - offset for x in axis_item.range])
    else:
        return

    for i, (_spacing, ticks) in enumerate(tick_levels):
        tick_length = abs(axis_item.style["tickLength"]) / ((i * 0.5) + 1.0)
        tick_pen = QtGui.QPen(axis_item.tickPen())
        color = QtGui.QColor(tick_pen.color())
        color.setAlpha(255 if i == 0 else 220)
        tick_pen.setColor(color)

        for value in ticks:
            pos = (value * scale) - offset
            if pos < visible_min or pos > visible_max:
                continue
            p1 = [pos, pos]
            p2 = [pos, pos]
            p1[axis_index] = tick_stop
            p2[axis_index] = tick_stop + tick_length * tick_dir
            tick_specs.append((tick_pen, pg.Point(p1), pg.Point(p2)))


def _manual_tick_levels(axis_item: pg.AxisItem, bounds):
    if axis_item._tickLevels is None:
        if axis_item.orientation == "left":
            span = (
                bounds.topRight() + pg.Point(-1.0, -1.0),
                bounds.bottomRight() + pg.Point(-1.0, 1.0),
            )
        elif axis_item.orientation == "bottom":
            span = (
                bounds.topLeft() + pg.Point(-1.0, 1.0),
                bounds.topRight() + pg.Point(1.0, 1.0),
            )
        else:
            return None
        points = list(map(axis_item.mapToDevice, span))
        if None in points:
            return None
        length_in_pixels = pg.Point(points[1] - points[0]).length()
        if length_in_pixels == 0:
            return None
        return axis_item.tickValues(axis_item.range[0], axis_item.range[1], length_in_pixels)

    tick_levels = []
    for level in axis_item._tickLevels:
        values = [value for value, _label in level]
        tick_levels.append((None, values))
    return tick_levels


class AbsoluteTimeAxis(pg.AxisItem):
    def __init__(self, orientation: str = "bottom") -> None:
        super().__init__(orientation=orientation)
        self._start_time: Optional[datetime] = None
        self._sample_rate: float = 1.0
        self.setStyle(
            tickFont=QtGui.QFont("Times New Roman", AXIS_TICK_FONT_SIZE_PT),
            tickTextOffset=8,
            tickLength=8,
        )
        self.setPen(pg.mkPen("k"))
        self.setTextPen(pg.mkPen("k"))

    def generateDrawSpecs(self, p):
        specs = super().generateDrawSpecs(p)
        if specs is None or self.grid is False or self.orientation != "bottom":
            return specs

        axis_spec, tick_specs, text_specs = specs
        bounds = self.mapRectFromParent(self.geometry())
        tick_levels = _manual_tick_levels(self, bounds)
        if tick_levels is None:
            return specs
        _append_axis_tick_stubs(self, tick_specs, bounds, tick_levels)
        return axis_spec, tick_specs, text_specs

    def set_context(self, start_time: datetime, sample_rate: float) -> None:
        self._start_time = start_time
        self._sample_rate = max(float(sample_rate), 1.0)
        self.picture = None
        self.update()

    def tickStrings(self, values, scale, spacing):
        if self._start_time is None:
            return [str(value) for value in values]

        labels = []
        for value in values:
            seconds = float(value) / self._sample_rate
            timestamp = self._start_time + timedelta(seconds=seconds)
            labels.append(timestamp.strftime("%H:%M:%S.%f")[:-3])
        return labels


class LogFrequencyAxis(pg.AxisItem):
    def __init__(self, orientation: str = "left") -> None:
        super().__init__(orientation=orientation)
        # For a left axis, positive tickLength draws ticks outward toward the labels.
        self.setStyle(
            tickFont=QtGui.QFont("Times New Roman", AXIS_TICK_FONT_SIZE_PT),
            tickTextOffset=8,
            tickLength=8,
        )
        self.setPen(pg.mkPen("k"))
        self.setTextPen(pg.mkPen("k"))

    def generateDrawSpecs(self, p):
        specs = super().generateDrawSpecs(p)
        if specs is None or self.grid is False or self.orientation != "left":
            return specs

        axis_spec, tick_specs, text_specs = specs
        bounds = self.mapRectFromParent(self.geometry())
        tick_levels = _manual_tick_levels(self, bounds)
        if tick_levels is None:
            return specs
        _append_axis_tick_stubs(self, tick_specs, bounds, tick_levels)
        return axis_spec, tick_specs, text_specs

    def tickStrings(self, values, scale, spacing):
        labels: list[str] = []
        for value in values:
            frequency = 10.0 ** float(value)
            if frequency >= 10000:
                labels.append(f"{frequency:.0f}")
            elif frequency >= 1000:
                labels.append(f"{frequency:.1f}")
            elif frequency >= 10:
                labels.append(f"{frequency:.0f}")
            else:
                labels.append(f"{frequency:.2f}")
        return labels


class LogPowerFrequencyAxis(pg.AxisItem):
    def __init__(self, orientation: str = "bottom") -> None:
        super().__init__(orientation=orientation)
        self.setStyle(
            tickFont=QtGui.QFont("Times New Roman", AXIS_TICK_FONT_SIZE_PT),
            tickTextOffset=8,
            tickLength=-8,
        )
        self.setPen(pg.mkPen("k"))
        self.setTextPen(pg.mkPen("k"))

    def generateDrawSpecs(self, p):
        specs = super().generateDrawSpecs(p)
        if specs is None or self.grid is False or self.orientation != "bottom":
            return specs

        axis_spec, tick_specs, text_specs = specs
        bounds = self.mapRectFromParent(self.geometry())
        tick_levels = _manual_tick_levels(self, bounds)
        if tick_levels is None:
            return specs
        _append_axis_tick_stubs(self, tick_specs, bounds, tick_levels)
        return axis_spec, tick_specs, text_specs

    def tickStrings(self, values, scale, spacing):
        labels: list[str] = []
        for value in values:
            frequency = 10.0 ** float(value)
            if abs(frequency - round(frequency)) < max(1e-9, frequency * 1e-9):
                labels.append(f"{int(round(frequency))}")
            else:
                labels.append(f"{frequency:g}")
        return labels


def _fallback_colormap(name: str) -> pg.ColorMap:
    key = str(name).strip().lower()
    if key == "grey":
        key = "gray"
    anchors = _FALLBACK_COLORMAPS.get(key, _FALLBACK_COLORMAPS["jet"])
    positions = [position for position, _color in anchors]
    colors = [color for _position, color in anchors]
    return pg.ColorMap(positions, colors, name=key)


def _coerce_colormap(candidate: object) -> pg.ColorMap | None:
    if isinstance(candidate, pg.ColorMap):
        return candidate
    return None


def _get_matplotlib_cmap(name: str):
    try:
        import matplotlib
    except Exception:
        return None

    try:
        registry = getattr(matplotlib, "colormaps", None)
        if registry is not None:
            return registry[name]
    except Exception:
        pass

    try:
        import matplotlib.cm as mpl_cm
        return mpl_cm.get_cmap(name)
    except Exception:
        return None


def _matplotlib_colormap(name: str) -> pg.ColorMap | None:
    col_map = _get_matplotlib_cmap(name)
    if col_map is None:
        return None

    color_map: pg.ColorMap | None = None
    if hasattr(col_map, "_segmentdata"):
        data = col_map._segmentdata
        if ("red" in data) and isinstance(data["red"], (Sequence, np.ndarray)):
            positions = set()
            for key in ("red", "green", "blue"):
                for item in data[key]:
                    positions.add(item[0])

            col_data = np.zeros((len(positions), 4), dtype=np.float64)
            col_data[:, -1] = sorted(positions)
            for index, key in enumerate(("red", "green", "blue")):
                channel_positions = np.zeros(len(data[key]), dtype=np.float64)
                channel_values = np.zeros(len(data[key]), dtype=np.float64)
                for item_index, item in enumerate(data[key]):
                    channel_positions[item_index] = item[0]
                    channel_values[item_index] = item[1]
                col_data[:, index] = np.interp(col_data[:, 3], channel_positions, channel_values)
            color_map = pg.ColorMap(pos=col_data[:, 3], color=(255 * col_data[:, :3]) + 0.5)
        elif ("red" in data) and isinstance(data["red"], Callable):
            col_data = np.zeros((64, 4), dtype=np.float64)
            col_data[:, -1] = np.linspace(0.0, 1.0, 64)
            for index, key in enumerate(("red", "green", "blue")):
                col_data[:, index] = np.clip(data[key](col_data[:, -1]), 0.0, 1.0)
            color_map = pg.ColorMap(pos=col_data[:, 3], color=(255 * col_data[:, :3]) + 0.5)
    elif hasattr(col_map, "colors"):
        try:
            from matplotlib.colors import to_rgba_array
            col_data = np.asarray(to_rgba_array(col_map.colors), dtype=np.float64)
        except Exception:
            col_data = np.asarray(col_map.colors, dtype=np.float64)
        if col_data.ndim == 2 and col_data.shape[0] > 0 and col_data.shape[1] >= 3:
            color_map = pg.ColorMap(
                name=name,
                pos=np.linspace(0.0, 1.0, col_data.shape[0]),
                color=(255 * col_data[:, :3]) + 0.5,
            )

    if color_map is not None:
        color_map.name = name
    return color_map


def create_colormap(name: str) -> pg.ColorMap:
    cmap_name = str(name).strip()
    if not cmap_name:
        cmap_name = "jet"

    fallback = _fallback_colormap(cmap_name)
    try:
        color_map = _coerce_colormap(pg.colormap.get(cmap_name, source="matplotlib"))
        if color_map is not None:
            return color_map
    except Exception:
        pass

    color_map = _matplotlib_colormap(cmap_name)
    if color_map is not None:
        return color_map

    try:
        color_map = _coerce_colormap(pg.colormap.get(cmap_name))
        if color_map is not None:
            return color_map
    except Exception:
        pass
    return fallback


def configure_plot_widget(plot_widget: pg.PlotWidget, left_label: str, bottom_label: str) -> None:
    axis_color = "#4B5563"
    text_color = "#1F2937"
    border_color = "#CBD5E1"
    bottom_label_text = "Time (hh:mm:ss.SSS)" if bottom_label == "Time" else bottom_label
    plot_widget.showGrid(x=True, y=True, alpha=0.22)
    plot_widget.setBackground("#FFFFFF")
    plot_widget.setLabel(
        "left",
        left_label,
        color=text_color,
        **{"font-family": "Times New Roman", "font-size": f"{AXIS_LABEL_FONT_SIZE_PT}pt"},
    )
    plot_widget.setLabel(
        "bottom",
        bottom_label_text,
        color=text_color,
        **{"font-family": "Times New Roman", "font-size": f"{AXIS_LABEL_FONT_SIZE_PT}pt"},
    )
    plot_item = plot_widget.getPlotItem()
    bottom_axis = plot_item.getAxis("bottom")
    bottom_axis.enableAutoSIPrefix(False)
    if bottom_label == "Time":
        bottom_axis.setHeight(TIME_AXIS_HEIGHT_PX)
    plot_item.getAxis("left").setStyle(
        tickFont=QtGui.QFont("Times New Roman", AXIS_TICK_FONT_SIZE_PT),
        tickTextOffset=8,
    )
    plot_item.getAxis("bottom").setStyle(
        tickFont=QtGui.QFont("Times New Roman", AXIS_TICK_FONT_SIZE_PT),
        tickTextOffset=8,
    )
    plot_item.getAxis("left").setPen(pg.mkPen(axis_color))
    plot_item.getAxis("left").setTextPen(pg.mkPen(text_color))
    plot_item.getAxis("bottom").setPen(pg.mkPen(axis_color))
    plot_item.getAxis("bottom").setTextPen(pg.mkPen(text_color))
    plot_item.getAxis("top").setPen(pg.mkPen(border_color))
    plot_item.getAxis("right").setPen(pg.mkPen(border_color))
    plot_item.getAxis("top").setTextPen(pg.mkPen(text_color))
    plot_item.getAxis("right").setTextPen(pg.mkPen(text_color))


def make_pen(color: str, width: int = 1):
    return pg.mkPen(color=color, width=width)
