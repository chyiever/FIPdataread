from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from scipy.signal import spectrogram


REPO_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = REPO_ROOT / "scripts"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from fipread.data_access import load_waveform  # noqa: E402


class AbsoluteTimeAxis(pg.AxisItem):
    def __init__(self, orientation: str = "bottom") -> None:
        super().__init__(orientation=orientation)
        self._start_time: datetime | None = None
        self.setStyle(tickFont=QtGui.QFont("Times New Roman", 11), tickTextOffset=8)
        self.setPen(pg.mkPen("k"))
        self.setTextPen(pg.mkPen("k"))

    def set_start_time(self, start_time: datetime) -> None:
        self._start_time = start_time

    def tickStrings(self, values, scale, spacing):
        if self._start_time is None:
            return [f"{value:.3f}" for value in values]

        labels: list[str] = []
        for value in values:
            timestamp = self._start_time + timedelta(seconds=float(value))
            labels.append(timestamp.strftime("%H:%M:%S.%f")[:-3])
        return labels


class LogFrequencyAxis(pg.AxisItem):
    def __init__(self, orientation: str = "left") -> None:
        super().__init__(orientation=orientation)
        self.setStyle(tickFont=QtGui.QFont("Times New Roman", 11), tickTextOffset=8)
        self.setPen(pg.mkPen("k"))
        self.setTextPen(pg.mkPen("k"))

    def tickStrings(self, values, scale, spacing):
        labels: list[str] = []
        for value in values:
            frequency = 10.0 ** float(value)
            if frequency >= 10_000:
                labels.append(f"{frequency:.0f}")
            elif frequency >= 1_000:
                labels.append(f"{frequency:.1f}")
            elif frequency >= 10:
                labels.append(f"{frequency:.0f}")
            else:
                labels.append(f"{frequency:.2f}")
        return labels


class SpectrogramViewBox(pg.ViewBox):
    def __init__(self) -> None:
        super().__init__()
        self.setMouseMode(self.RectMode)

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_R, QtCore.Qt.Key_A):
            self.enableAutoRange()
            event.accept()
            return
        super().keyPressEvent(event)


def validate_inputs(
    input_path: Path,
    window_seconds: float,
    overlap: float,
    log_bins: int,
    min_frequency: float,
    sample_rate_override: float | None,
    vmin: float | None,
    vmax: float | None,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if window_seconds <= 0:
        raise ValueError("--window-seconds must be > 0.")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("--overlap must be in [0, 1).")
    if log_bins < 32:
        raise ValueError("--log-bins must be >= 32.")
    if min_frequency <= 0:
        raise ValueError("--min-frequency must be > 0.")
    if sample_rate_override is not None and sample_rate_override <= 0:
        raise ValueError("sample_rate_override must be > 0 when provided.")
    if vmin is not None and vmax is not None and vmin >= vmax:
        raise ValueError("vmin must be smaller than vmax.")


def compute_psd_spectrogram(
    values: np.ndarray,
    sample_rate: float,
    window_seconds: float,
    overlap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal = np.asarray(values, dtype=np.float64).reshape(-1)
    if signal.size < 2:
        raise ValueError("Input waveform is too short.")

    nperseg = max(8, int(round(sample_rate * window_seconds)))
    nperseg = min(nperseg, signal.size)
    if nperseg < 8:
        raise ValueError("Window is too short after matching the signal length.")

    noverlap = min(int(round(nperseg * overlap)), nperseg - 1)
    freqs, times, psd = spectrogram(
        signal,
        fs=float(sample_rate),
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="linear",
        scaling="density",
        mode="psd",
        return_onesided=True,
    )
    psd_db = 10.0 * np.log10(np.maximum(psd, np.finfo(np.float64).tiny))
    return freqs, times, psd_db


def rebin_to_log_frequency(
    freqs: np.ndarray,
    psd_db: np.ndarray,
    min_frequency: float,
    log_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid = freqs >= max(float(min_frequency), np.nextafter(0.0, 1.0))
    valid &= freqs > 0
    filtered_freqs = freqs[valid]
    filtered_psd = psd_db[valid, :]
    if filtered_freqs.size == 0:
        raise ValueError("No positive frequencies are available for plotting.")

    if filtered_freqs.size <= log_bins:
        return filtered_freqs, filtered_psd

    edges = np.geomspace(filtered_freqs[0], filtered_freqs[-1], log_bins + 1)
    indices = np.searchsorted(filtered_freqs, edges)

    rebinned = np.empty((log_bins, filtered_psd.shape[1]), dtype=np.float64)
    centers = np.empty(log_bins, dtype=np.float64)
    last_row = filtered_psd[0, :]
    last_freq = filtered_freqs[0]

    for idx in range(log_bins):
        start = min(indices[idx], filtered_freqs.size - 1)
        stop = min(indices[idx + 1], filtered_freqs.size)
        if stop <= start:
            rebinned[idx, :] = last_row
            centers[idx] = last_freq
            continue

        rebinned[idx, :] = filtered_psd[start:stop, :].mean(axis=0)
        centers[idx] = math.sqrt(filtered_freqs[start] * filtered_freqs[stop - 1])
        last_row = rebinned[idx, :]
        last_freq = centers[idx]

    return centers, rebinned


def create_colormap(colormap_name: str) -> pg.ColorMap:
    try:
        return pg.colormap.get(colormap_name, source="matplotlib")
    except Exception:
        pass

    try:
        return pg.colormap.get(colormap_name)
    except Exception as exc:
        raise ValueError(
            f"Unsupported colormap '{colormap_name}'. Try names like "
            f"'jet', 'inferno', 'viridis', 'plasma', 'magma', or 'turbo'."
        ) from exc


def resolve_levels(psd_db: np.ndarray, vmin: float | None, vmax: float | None) -> tuple[float, float]:
    data_min = float(np.nanmin(psd_db))
    data_max = float(np.nanmax(psd_db))

    low = data_min if vmin is None else float(vmin)
    high = data_max if vmax is None else float(vmax)
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("vmin/vmax must be finite numbers.")
    if low >= high:
        raise ValueError("Resolved color levels are invalid: vmin must be smaller than vmax.")
    return low, high


def show_time_frequency(
    start_time: datetime,
    times: np.ndarray,
    log_freqs: np.ndarray,
    psd_db: np.ndarray,
    source_name: str,
    colormap_name: str,
    vmin: float | None,
    vmax: float | None,
) -> int:
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    pg.setConfigOptions(antialias=False, background="w", foreground="k")

    time_axis = AbsoluteTimeAxis("bottom")
    time_axis.set_start_time(start_time)
    freq_axis = LogFrequencyAxis("left")
    color_map = create_colormap(colormap_name)
    levels = resolve_levels(psd_db, vmin=vmin, vmax=vmax)

    graphics = pg.GraphicsLayoutWidget(show=True, title=f"Time-Frequency PSD - {source_name}")
    graphics.resize(1400, 900)

    view_box = SpectrogramViewBox()
    view_box.setMouseEnabled(x=True, y=True)
    view_box.enableAutoRange()
    view_box.setAutoVisible(x=True, y=True)

    plot_item = graphics.addPlot(
        row=0,
        col=0,
        viewBox=view_box,
        axisItems={"bottom": time_axis, "left": freq_axis},
    )
    plot_item.setLabel("bottom", "Time", units="")
    plot_item.setLabel("left", "Frequency (log10 Hz)", units="")
    plot_item.showGrid(x=True, y=True, alpha=0.2)
    plot_item.setMenuEnabled(True)
    plot_item.setClipToView(True)

    image = pg.ImageItem(psd_db, axisOrder="row-major")
    image.setColorMap(color_map)
    image.setLevels(levels)
    plot_item.addItem(image)

    x0 = float(times[0]) if times.size else 0.0
    x1 = float(times[-1]) if times.size else 0.0
    if times.size > 1:
        dt = float(np.median(np.diff(times)))
    else:
        dt = 0.0
    if log_freqs.size > 1:
        log_step = float(np.median(np.diff(np.log10(log_freqs))))
    else:
        log_step = 0.0

    y0 = float(np.log10(log_freqs[0]))
    y1 = float(np.log10(log_freqs[-1]))
    image.setRect(QtCore.QRectF(x0 - 0.5 * dt, y0 - 0.5 * log_step, (x1 - x0) + dt, (y1 - y0) + log_step))
    plot_item.setLimits(xMin=x0 - dt, xMax=x1 + dt, yMin=y0 - log_step, yMax=y1 + log_step)
    plot_item.setXRange(x0, x1 if x1 > x0 else x0 + 1e-6, padding=0.0)
    plot_item.setYRange(y0, y1 if y1 > y0 else y0 + 1e-6, padding=0.0)

    histogram = pg.HistogramLUTItem(image=image)
    histogram.gradient.setColorMap(color_map)
    histogram.setLevels(*levels)
    graphics.addItem(histogram, row=0, col=1)

    graphics.setWindowTitle(f"Time-Frequency PSD - {source_name}")
    plot_item.setTitle("Left drag: rectangular zoom | Middle/right drag: pan | Wheel: zoom | R/A: reset")

    if owns_app:
        return app.exec_()
    return 0


def main() -> int:
    input_path = Path(r"E:\codes\FIPread\data\0000011-500K-20260324T033931.619.tdms")
    window_seconds = 0.01
    overlap = 0.5
    sample_rate_override: float | None = None
    log_bins = 512
    min_frequency = 1.0
    colormap = "jet"
    vmin: float | None = -130.0
    vmax: float | None = -60.0

    validate_inputs(
        input_path=input_path,
        window_seconds=window_seconds,
        overlap=overlap,
        log_bins=log_bins,
        min_frequency=min_frequency,
        sample_rate_override=sample_rate_override,
        vmin=vmin,
        vmax=vmax,
    )

    loaded = load_waveform(input_path)
    sample_rate = loaded.sample_rate if sample_rate_override is None else float(sample_rate_override)
    freqs, times, psd_db = compute_psd_spectrogram(
        values=loaded.phase_data,
        sample_rate=sample_rate,
        window_seconds=window_seconds,
        overlap=overlap,
    )
    log_freqs, rebinned_psd = rebin_to_log_frequency(
        freqs=freqs,
        psd_db=psd_db,
        min_frequency=min_frequency,
        log_bins=log_bins,
    )

    print(f"file={loaded.path}")
    print(f"sample_rate_hz={sample_rate:.3f}")
    print(f"waveform_samples={loaded.phase_data.size}")
    print(f"time_slices={times.size}")
    print(f"frequency_bins_raw={freqs.size}")
    print(f"frequency_bins_display={log_freqs.size}")
    print(f"start_time={loaded.start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    print(f"colormap={colormap}")
    print(f"vmin={vmin}")
    print(f"vmax={vmax}")

    return show_time_frequency(
        start_time=loaded.start_time,
        times=times,
        log_freqs=log_freqs,
        psd_db=rebinned_psd,
        source_name=loaded.path.name,
        colormap_name=colormap,
        vmin=vmin,
        vmax=vmax,
    )


if __name__ == "__main__":
    raise SystemExit(main())
