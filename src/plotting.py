from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from PyQt5 import QtGui
import pyqtgraph as pg


class AbsoluteTimeAxis(pg.AxisItem):
    def __init__(self, orientation: str = "bottom") -> None:
        super().__init__(orientation=orientation)
        self._start_time: Optional[datetime] = None
        self._sample_rate: float = 1.0
        self.setStyle(tickFont=QtGui.QFont("Times New Roman", 11), tickTextOffset=8)
        self.setPen(pg.mkPen("k"))
        self.setTextPen(pg.mkPen("k"))

    def set_context(self, start_time: datetime, sample_rate: float) -> None:
        self._start_time = start_time
        self._sample_rate = max(float(sample_rate), 1.0)

    def tickStrings(self, values, scale, spacing):
        if self._start_time is None:
            return [str(value) for value in values]

        labels = []
        for value in values:
            seconds = float(value) / self._sample_rate
            timestamp = self._start_time + timedelta(seconds=seconds)
            labels.append(timestamp.strftime("%H:%M:%S.%f")[:-3])
        return labels


def configure_plot_widget(plot_widget: pg.PlotWidget, left_label: str, bottom_label: str) -> None:
    axis_color = "#4B5563"
    text_color = "#1F2937"
    border_color = "#CBD5E1"
    plot_widget.showGrid(x=True, y=True, alpha=0.22)
    plot_widget.setBackground("#FFFFFF")
    plot_widget.setLabel("left", left_label, color=text_color, **{"font-family": "Times New Roman", "font-size": "13pt"})
    plot_widget.setLabel("bottom", bottom_label, color=text_color, **{"font-family": "Times New Roman", "font-size": "13pt"})
    plot_item = plot_widget.getPlotItem()
    plot_item.getAxis("left").setStyle(tickFont=QtGui.QFont("Times New Roman", 11), tickTextOffset=8)
    plot_item.getAxis("bottom").setStyle(tickFont=QtGui.QFont("Times New Roman", 11), tickTextOffset=8)
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
