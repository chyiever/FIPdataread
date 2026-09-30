"""Application entry point.

Puts ``src`` on ``sys.path`` and starts the Qt event loop around ``MainWindow``.
"""

from __future__ import annotations


import sys
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from main_window import MainWindow


def main() -> int:
    """Entry point: configure high-DPI Qt, build MainWindow and run the app.

    Enables high-DPI scaling before the QApplication is created so that point-based
    fonts and widget geometry are scaled by the same device-pixel ratio, which keeps
    button labels from being clipped on high-scaling displays.
    """

    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("FIPread")
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
