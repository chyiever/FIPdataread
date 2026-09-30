from __future__ import annotations

import os
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import load as joblib_load
from scipy.signal import butter, get_window, sosfiltfilt, welch
from scipy import signal as scipy_signal
from scipy import stats
from sklearn.pipeline import Pipeline

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


def prepare_audio_waveform(
    values: np.ndarray,
    sample_rate: float,
    downsample_factor: int,
    *,
    target_peak: float = 0.95,
) -> tuple[np.ndarray, int]:
    signal = np.asarray(values, dtype=np.float64).reshape(-1)
    if signal.size == 0:
        raise ValueError("Audio source waveform is empty.")
    if sample_rate <= 0.0:
        raise ValueError("Sample rate must be greater than 0.")

    factor = max(1, int(downsample_factor))
    centered = signal - float(np.mean(signal))
    if factor > 1:
        centered = scipy_signal.resample_poly(centered, up=1, down=factor)

    audio_sample_rate = max(1, int(round(sample_rate / factor)))
    peak = float(np.percentile(np.abs(centered), 99.5)) if centered.size else 0.0
    if peak <= 0.0:
        peak = float(np.max(np.abs(centered))) if centered.size else 0.0
    if peak <= 0.0:
        return np.zeros(centered.shape, dtype=np.int16), audio_sample_rate

    scaled = centered * (float(target_peak) / peak)
    clipped = np.clip(scaled, -1.0, 1.0)
    pcm16 = np.round(clipped * 32767.0).astype(np.int16)
    return pcm16, audio_sample_rate



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


@dataclass(frozen=True)
class SlidingWindowSVMPredictor:
    model: Pipeline
    selected_features: tuple[str, ...]
    energy_bands: tuple[tuple[float, float], ...]
    stat_bands: tuple[tuple[float, float], ...]


def _band_name(low_hz: float, high_hz: float) -> str:
    return f"{int(low_hz / 1000)}k_{int(high_hz / 1000)}k"


def _safe_float(value: float) -> float:
    result = float(value)
    if not np.isfinite(result):
        return 0.0
    return result


def _fft_power(signal_data: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.fft.rfft(signal_data)
    freqs = np.fft.rfftfreq(signal_data.size, d=1.0 / sample_rate)
    power = np.abs(spectrum) ** 2
    return freqs, power


def _band_energy_from_spectrum(freqs: np.ndarray, power: np.ndarray, low_hz: float, high_hz: float) -> float:
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        return 0.0
    return float(power[mask].sum())


def _bandpass_filter(
    signal_data: np.ndarray,
    sample_rate: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    nyquist = 0.5 * sample_rate
    low = max(low_hz / nyquist, 1e-6)
    high = min(high_hz / nyquist, 0.999999)
    if not (0 < low < high < 1):
        raise ValueError(f"Invalid band limits: {(low_hz, high_hz)} for sample rate {sample_rate}")
    sos = scipy_signal.butter(order, [low, high], btype="bandpass", output="sos")
    return scipy_signal.sosfiltfilt(sos, signal_data)


def _highpass_filter(
    signal_data: np.ndarray,
    sample_rate: float,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    nyquist = 0.5 * sample_rate
    cutoff = max(cutoff_hz / nyquist, 1e-6)
    if not (0 < cutoff < 1):
        raise ValueError(f"Invalid high-pass cutoff: {cutoff_hz} for sample rate {sample_rate}")
    sos = scipy_signal.butter(order, cutoff, btype="highpass", output="sos")
    return scipy_signal.sosfiltfilt(sos, signal_data)


def _compute_chunk_size(window_samples: int, selected_feature_count: int) -> int:
    target_bytes = 32 * 1024 * 1024
    per_window_bytes = max(window_samples * 8 * (1 + selected_feature_count // 8), window_samples * 16)
    return max(8, min(512, target_bytes // max(per_window_bytes, 1)))


def _make_feature_store(window_count: int, feature_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {
        feature_name: np.zeros(window_count, dtype=np.float64)
        for feature_name in feature_names
    }


def _iter_window_chunks(
    window_view: np.ndarray,
    chunk_size: int,
):
    for chunk_start in range(0, window_view.shape[0], chunk_size):
        chunk_stop = min(window_view.shape[0], chunk_start + chunk_size)
        yield chunk_start, chunk_stop, np.asarray(window_view[chunk_start:chunk_stop], dtype=np.float64)


def _assign_energy_features(
    store: dict[str, np.ndarray],
    chunk_slice: slice,
    power: np.ndarray,
    freqs: np.ndarray,
    energy_bands: tuple[tuple[float, float], ...],
    energy_ratio_range: tuple[float, float],
) -> None:
    floor = np.finfo(np.float64).tiny
    ratio_range_label = _band_name(*energy_ratio_range)
    total_mask = (freqs >= energy_ratio_range[0]) & (freqs < energy_ratio_range[1])
    total_energy = power[:, total_mask].sum(axis=1) if np.any(total_mask) else np.zeros(power.shape[0], dtype=np.float64)
    total_energy_safe = np.maximum(total_energy, floor)
    total_feature_name = f"energy_total_{ratio_range_label}"
    if total_feature_name in store:
        store[total_feature_name][chunk_slice] = total_energy

    for low_hz, high_hz in energy_bands:
        band_label = _band_name(low_hz, high_hz)
        band_mask = (freqs >= low_hz) & (freqs < high_hz)
        energy = power[:, band_mask].sum(axis=1) if np.any(band_mask) else np.zeros(power.shape[0], dtype=np.float64)
        energy = np.asarray(energy, dtype=np.float64)
        energy_name = f"energy_{band_label}"
        if energy_name in store:
            store[energy_name][chunk_slice] = energy
        ratio_name = f"energy_ratio_{band_label}_within_{ratio_range_label}"
        if ratio_name in store:
            store[ratio_name][chunk_slice] = energy / total_energy_safe


def _assign_stat_features(
    store: dict[str, np.ndarray],
    chunk_slice: slice,
    windows: np.ndarray,
    sample_rate: float,
    band_label: str,
) -> None:
    centered = windows - windows.mean(axis=1, keepdims=True)
    zcr = np.count_nonzero(np.signbit(centered[:, 1:]) != np.signbit(centered[:, :-1]), axis=1)
    zcr = zcr.astype(np.float64) / max(windows.shape[1] - 1, 1)
    kurtosis = np.asarray(stats.kurtosis(windows, axis=1, fisher=False, bias=False), dtype=np.float64)

    spectrum = np.fft.rfft(windows, axis=1)
    power = np.abs(spectrum) ** 2
    amplitude = np.sqrt(power)
    freqs = np.fft.rfftfreq(windows.shape[1], d=1.0 / sample_rate)
    total_power = power.sum(axis=1)
    total_power_safe = np.maximum(total_power, np.finfo(np.float64).tiny)

    centroid = (power * freqs[np.newaxis, :]).sum(axis=1) / total_power_safe
    peak_idx = np.argmax(amplitude, axis=1)
    peak_frequency = freqs[peak_idx]
    peak_amplitude = np.take_along_axis(amplitude, peak_idx[:, np.newaxis], axis=1).ravel()
    power_safe = np.maximum(power, np.finfo(np.float64).tiny)
    flatness = np.exp(np.mean(np.log(power_safe), axis=1)) / np.mean(power_safe, axis=1)

    stat_values = {
        f"kurtosis_{band_label}": np.nan_to_num(kurtosis, nan=0.0, posinf=0.0, neginf=0.0),
        f"zcr_{band_label}": np.nan_to_num(zcr, nan=0.0, posinf=0.0, neginf=0.0),
        f"spectral_centroid_hz_{band_label}": np.nan_to_num(centroid, nan=0.0, posinf=0.0, neginf=0.0),
        f"peak_frequency_hz_{band_label}": np.nan_to_num(peak_frequency, nan=0.0, posinf=0.0, neginf=0.0),
        f"peak_amplitude_{band_label}": np.nan_to_num(peak_amplitude, nan=0.0, posinf=0.0, neginf=0.0),
        f"spectral_flatness_{band_label}": np.nan_to_num(flatness, nan=0.0, posinf=0.0, neginf=0.0),
    }
    for feature_name, values in stat_values.items():
        if feature_name in store:
            store[feature_name][chunk_slice] = values


def _compute_stat_band_feature_block(
    signal_data: np.ndarray,
    sample_rate: float,
    low_hz: float,
    high_hz: float,
    window_samples: int,
    hop_samples: int,
    chunk_size: int,
    selected_features: tuple[str, ...],
    active_indices: np.ndarray,
) -> tuple[str, dict[str, np.ndarray]]:
    band_label = _band_name(low_hz, high_hz)
    feature_store = _make_feature_store(
        active_indices.size,
        tuple(feature_name for feature_name in selected_features if feature_name.endswith(f"_{band_label}")),
    )
    if not feature_store:
        return band_label, {}

    filtered_signal = _bandpass_filter(signal_data, sample_rate, low_hz, high_hz)
    all_filtered_windows = np.lib.stride_tricks.sliding_window_view(filtered_signal, window_samples)[::hop_samples]
    filtered_window_view = all_filtered_windows[active_indices]
    for chunk_start, chunk_stop, filtered_windows in _iter_window_chunks(filtered_window_view, chunk_size):
        _assign_stat_features(
            feature_store,
            slice(chunk_start, chunk_stop),
            filtered_windows,
            sample_rate,
            band_label,
        )
    return band_label, feature_store


@lru_cache(maxsize=4)
def load_sliding_window_svm_predictor(model_directory: str) -> SlidingWindowSVMPredictor:
    model_dir = Path(model_directory)
    metadata_path = model_dir / "svm_model_metadata.json"
    model_path = model_dir / "svm_model.joblib"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    model = joblib_load(model_path)
    return SlidingWindowSVMPredictor(
        model=model,
        selected_features=tuple(metadata["selected_features"]),
        energy_bands=tuple(tuple(float(value) for value in band) for band in metadata["energy_bands"]),
        stat_bands=tuple(tuple(float(value) for value in band) for band in metadata["stat_bands"]),
    )


def compute_short_time_svm_predictions(
    values: np.ndarray,
    sample_rate: float,
    *,
    predictor: SlidingWindowSVMPredictor,
    window_seconds: float = 0.04,
    hop_ratio: float = 0.5,
    gate_highpass_hz: float = 20000.0,
    gate_peak_threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(values, dtype=np.float64)
    if signal.size == 0 or sample_rate <= 0.0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    window_samples = max(2, int(round(window_seconds * sample_rate)))
    if window_samples > signal.size:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    hop_samples = max(1, int(round(window_samples * hop_ratio)))
    starts = np.arange(0, signal.size - window_samples + 1, hop_samples, dtype=np.int64)
    if starts.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    centers = starts.astype(np.float64) + (window_samples * 0.5)
    predictions = np.zeros(starts.size, dtype=np.float64)
    chunk_size = _compute_chunk_size(window_samples, len(predictor.selected_features))
    ratio_range = (1000.0, 60000.0)
    raw_window_view = np.lib.stride_tricks.sliding_window_view(signal, window_samples)[::hop_samples]
    gate_signal = _highpass_filter(signal, sample_rate, gate_highpass_hz)
    gate_window_view = np.lib.stride_tricks.sliding_window_view(gate_signal, window_samples)[::hop_samples]
    gate_peak = np.max(np.abs(gate_window_view), axis=1)
    active_indices = np.flatnonzero(gate_peak > gate_peak_threshold)
    if active_indices.size == 0:
        return centers, predictions

    active_raw_window_view = raw_window_view[active_indices]
    feature_store = _make_feature_store(active_indices.size, predictor.selected_features)
    raw_freqs = np.fft.rfftfreq(window_samples, d=1.0 / sample_rate)

    for chunk_start, chunk_stop, raw_windows in _iter_window_chunks(active_raw_window_view, chunk_size):
        chunk_slice = slice(chunk_start, chunk_stop)
        raw_spectrum = np.fft.rfft(raw_windows, axis=1)
        raw_power = np.abs(raw_spectrum) ** 2
        _assign_energy_features(
            feature_store,
            chunk_slice,
            raw_power,
            raw_freqs,
            predictor.energy_bands,
            ratio_range,
        )

    max_workers = min(len(predictor.stat_bands), max(1, os.cpu_count() or 1))
    if max_workers <= 1:
        stat_results = [
            _compute_stat_band_feature_block(
                signal,
                sample_rate,
                low_hz,
                high_hz,
                window_samples,
                hop_samples,
                chunk_size,
                predictor.selected_features,
                active_indices,
            )
            for low_hz, high_hz in predictor.stat_bands
        ]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _compute_stat_band_feature_block,
                    signal,
                    sample_rate,
                    low_hz,
                    high_hz,
                    window_samples,
                    hop_samples,
                    chunk_size,
                    predictor.selected_features,
                    active_indices,
                )
                for low_hz, high_hz in predictor.stat_bands
            ]
            stat_results = [future.result() for future in futures]

    for _band_label, stat_feature_store in stat_results:
        for feature_name, values in stat_feature_store.items():
            feature_store[feature_name] = values

    feature_frame = pd.DataFrame(
        {feature_name: feature_store[feature_name] for feature_name in predictor.selected_features}
    )
    predictions[active_indices] = predictor.model.predict(feature_frame).astype(np.float64, copy=False)
    return centers, predictions
