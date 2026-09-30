from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import butter, get_window, sosfiltfilt, welch

from .models import FilterMode


def validate_filter(
    enabled: bool,
    mode: FilterMode,
    sample_rate: float,
    low_cut_hz: float,
    high_cut_hz: float,
) -> tuple[bool, Optional[str]]:
    if not enabled:
        return True, None

    nyquist = sample_rate / 2.0
    if mode == FilterMode.BANDPASS:
        if low_cut_hz <= 0 or high_cut_hz <= 0:
            return False, "Band-pass cutoff values must be > 0."
        if low_cut_hz >= high_cut_hz:
            return False, "Band-pass low cutoff must be lower than high cutoff."
        if high_cut_hz >= nyquist:
            return False, f"High cutoff must be lower than Nyquist ({nyquist:.1f} Hz)."
    elif mode == FilterMode.HIGHPASS:
        if low_cut_hz <= 0:
            return False, "High-pass cutoff must be > 0."
        if low_cut_hz >= nyquist:
            return False, f"High-pass cutoff must be lower than Nyquist ({nyquist:.1f} Hz)."
    elif mode == FilterMode.LOWPASS:
        if high_cut_hz <= 0:
            return False, "Low-pass cutoff must be > 0."
        if high_cut_hz >= nyquist:
            return False, f"Low-pass cutoff must be lower than Nyquist ({nyquist:.1f} Hz)."
    return True, None


def apply_display_filter(
    values: np.ndarray,
    sample_rate: float,
    enabled: bool,
    mode: FilterMode,
    low_cut_hz: float,
    high_cut_hz: float,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.size == 0 or not enabled:
        return result

    nyquist = sample_rate / 2.0
    if mode == FilterMode.BANDPASS:
        sos = butter(
            N=4,
            Wn=[low_cut_hz / nyquist, high_cut_hz / nyquist],
            btype="bandpass",
            output="sos",
        )
    elif mode == FilterMode.HIGHPASS:
        sos = butter(
            N=4,
            Wn=low_cut_hz / nyquist,
            btype="highpass",
            output="sos",
        )
    else:
        sos = butter(
            N=4,
            Wn=high_cut_hz / nyquist,
            btype="lowpass",
            output="sos",
        )

    return sosfiltfilt(sos, result)


def compute_window_psd(values: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(values, dtype=np.float64)
    if signal.size < 2:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    nperseg = signal.size
    noverlap = nperseg // 2
    freqs, psd = welch(
        signal,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="linear",
        scaling="density",
        return_onesided=True,
    )
    floor = np.finfo(np.float64).tiny
    psd_db = 10.0 * np.log10(np.maximum(psd, floor))
    return freqs, psd_db



def compute_short_time_energy_ratio(
    values: np.ndarray,
    sample_rate: float,
    *,
    numerator_low_hz: float,
    numerator_high_hz: float,
    denominator_low_hz: float,
    denominator_high_hz: float,
    window_seconds: float,
    hop_ratio: float,
    amplitude_threshold: float,
    gate_values: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(values, dtype=np.float64)
    gate_signal = signal if gate_values is None else np.asarray(gate_values, dtype=np.float64)
    if signal.size == 0 or sample_rate <= 0.0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    if gate_signal.size != signal.size:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    window_samples = max(2, int(round(window_seconds * sample_rate)))
    if window_samples > signal.size:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    hop_samples = max(1, int(round(window_samples * hop_ratio)))
    starts = np.arange(0, signal.size - window_samples + 1, hop_samples, dtype=np.int64)
    if starts.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    freqs = np.fft.rfftfreq(window_samples, d=1.0 / sample_rate)
    numerator_mask = (freqs >= numerator_low_hz) & (freqs <= numerator_high_hz)
    denominator_mask = (freqs >= denominator_low_hz) & (freqs <= denominator_high_hz)
    if not np.any(numerator_mask) or not np.any(denominator_mask):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    numerator_bandwidth = max(numerator_high_hz - numerator_low_hz, np.finfo(np.float64).eps)
    denominator_bandwidth = max(denominator_high_hz - denominator_low_hz, np.finfo(np.float64).eps)
    window = get_window('hann', window_samples, fftbins=True).astype(np.float64, copy=False)
    floor = np.finfo(np.float64).tiny
    centers = np.empty(starts.size, dtype=np.float64)
    ratios_db = np.empty(starts.size, dtype=np.float64)

    for index, start in enumerate(starts):
        stop = int(start + window_samples)
        segment = signal[int(start):stop]
        gate_segment = gate_signal[int(start):stop]
        centers[index] = start + (window_samples * 0.5)
        if segment.size < window_samples or gate_segment.size < window_samples:
            ratios_db[index] = 0.0
            continue
        if float(np.max(np.abs(gate_segment))) < amplitude_threshold:
            ratios_db[index] = 0.0
            continue

        spectrum = np.fft.rfft(segment * window)
        power = np.abs(spectrum) ** 2
        numerator_energy = float(np.sum(power[numerator_mask]))
        denominator_energy = float(np.sum(power[denominator_mask]))
        numerator_density = numerator_energy / numerator_bandwidth
        denominator_density = denominator_energy / denominator_bandwidth
        ratio = numerator_density / max(denominator_density, floor)
        ratios_db[index] = 10.0 * np.log10(max(ratio, floor))

    return centers, ratios_db
