"""Splitter sizing, t-f tab alignment and window resize handling.

Part of the ``MainWindow`` mixin set; see ``main_window`` for the full table of
responsibilities.
"""

from PyQt5 import QtCore
from PyQt5 import QtGui
from constants import FEATURE_MODE_NONE


class LayoutMixin:
    """Splitter sizing, t-f tab alignment and window resize handling.

    Qt has to have performed a layout pass before the real plot heights can be
    measured, which is why these methods are driven from ``resizeEvent`` and from
    the tab-change handlers rather than from ``_build_ui``.
    """

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Recompute plot splitter sizes whenever the window is resized.
        """

        super().resizeEvent(event)
        QtCore.QTimer.singleShot(0, self._apply_right_plot_splitter_sizes)


    def _update_time_tf_alignment_for_tab(self, index: int) -> None:
        """Keep the top time plot aligned with the t-f plot.

        The t-f page reserves room for its colour bar, so its drawing area is narrower
        than the top panel's. A right-edge spacer is widened by exactly that amount on
        the t-f tab so both time axes line up; on ``1D Curve`` it is collapsed to zero to
        reclaim the full width.
        """

        if int(index) == 1:
            self.time_right_spacer.setMinimumWidth(self._tf_time_axis_compensation)
            self.time_right_spacer.setMaximumWidth(self._tf_time_axis_compensation)
        else:
            self.time_right_spacer.setMinimumWidth(0)
            self.time_right_spacer.setMaximumWidth(0)
        self._apply_right_plot_splitter_sizes()
        QtCore.QTimer.singleShot(0, self._apply_right_plot_splitter_sizes)
        self.analysis_tabs.updateGeometry()
        self.tf_plot.updateGeometry()
        self.tf_histogram.updateGeometry()
        if int(index) == 1:
            self._sync_tf_x_from_time()


    def _update_curve_splitter_for_feature_mode(self) -> None:
        """Show or collapse Plot 2 depending on the selected mode.
        """

        splitter = getattr(self, "_curve_splitter", None)
        if splitter is None:
            return
        if self._current_feature_mode() == FEATURE_MODE_NONE:
            splitter.setSizes([0, 540])
        else:
            splitter.setSizes([270, 540])
        self._apply_right_plot_splitter_sizes()


    def _apply_right_plot_splitter_sizes(self) -> None:
        """Size Plot 1 / Plot 2 / PSD in a 1:1:2 ratio.

        Heights are derived from Qt's *measured* layout overhead rather than fixed
        estimates, so the ratio holds across window sizes, DPI settings and fonts. The
        measurement is read fresh because the previous fixed-value approach left the two
        time-domain plots at different heights on some displays.
        """

        right_panel = getattr(self, "_right_panel_splitter", None)
        if right_panel is None:
            return
        if self.analysis_tabs.currentIndex() == 1:
            right_panel.setSizes([320, 640])
        elif self._current_feature_mode() == FEATURE_MODE_NONE:
            curve_splitter = getattr(self, "_curve_splitter", None)
            if curve_splitter is not None:
                curve_splitter.setSizes([0, max(1, curve_splitter.height())])
            right_panel.setSizes([320, 640])
        else:
            curve_splitter = getattr(self, "_curve_splitter", None)
            if curve_splitter is None:
                return
            total = max(1, right_panel.height())
            time_overhead = self._measure_time_panel_overhead()
            analysis_overhead = max(0, self.analysis_tabs.height() - curve_splitter.height())
            right_handle = max(0, right_panel.handleWidth())
            available = total - time_overhead - analysis_overhead - right_handle
            if available <= 0:
                right_panel.setSizes([320, 640])
                curve_splitter.setSizes([270, 540])
                return
            plot_unit = max(1, int(round(available / 4.0)))
            top_panel = time_overhead + plot_unit
            analysis_panel = analysis_overhead + plot_unit * 3
            right_panel.setSizes([max(1, int(top_panel)), max(1, int(analysis_panel))])
            curve_splitter.setSizes([plot_unit, plot_unit * 2])


    def _measure_time_panel_overhead(self) -> int:
        """Measure the non-plot height of the top panel from the live layout.
        """

        layout = getattr(self, "_time_column_layout", None)
        if layout is None:
            return 0
        plot_index = layout.indexOf(self.time_plot)
        overhead = 0
        for index in range(layout.count()):
            if index == plot_index:
                continue
            item = layout.itemAt(index)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                overhead += max(widget.height(), widget.sizeHint().height())
            else:
                overhead += max(item.geometry().height(), item.sizeHint().height())
        overhead += max(0, layout.spacing()) * max(0, layout.count() - 1)
        margins = layout.contentsMargins()
        overhead += margins.top() + margins.bottom()
        return max(0, int(overhead))


    @staticmethod
    def _states_close(
        left: tuple[tuple[float, float], tuple[float, float]],
        right: tuple[tuple[float, float], tuple[float, float]],
    ) -> bool:
        """Return True when two view states are effectively identical.

        Used by the view-history code to avoid pushing a duplicate state when the
        view barely moved, and to drop states that are indistinguishable once
        repainted.
        """

        return all(
            abs(a - b) < 1e-6
            for pair_left, pair_right in zip(left, right)
            for a, b in zip(pair_left, pair_right)
        )


    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Stop media playback and remove the temp audio file before closing.

        Thread teardown is handled by each panel's own finished-handler; this only
        deals with the audio resources that live as long as the window does.
        """

        self._audio_player.stop()
        self._clear_audio_temp_path()
        super().closeEvent(event)
