from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import wavfile

from models import FileRecord, LoadedWaveform, PagedFiles, SortField


TIME_TOKEN_RE = re.compile(r"(?P<stamp>\d{8}T\d{6}(?:\.\d{1,6})?)")
SAMPLE_RATE_TOKEN_RE = re.compile(r"-(?P<rate>\d+(?:\.\d+)?)K-", re.IGNORECASE)
ARRIVAL_TIME_TOKEN_RE = re.compile(r"(?P<stamp>\d{14}(?:\.\d{1,6})?)")
SUPPORTED_SUFFIXES = {".npz", ".tdms"}


def format_start_time_token(start_time: datetime) -> str:
    return start_time.strftime("%Y%m%dT%H%M%S.%f")[:-3]


def format_arrival_time_token(arrival_time: datetime) -> str:
    # Keep 0.1 ms precision (4 digits after decimal point).
    return arrival_time.strftime("%Y%m%d%H%M%S.%f")[:-2]


def format_sample_rate_token(sample_rate: float) -> str:
    rate_khz = float(sample_rate) / 1_000.0
    if abs(rate_khz - round(rate_khz)) < 1e-9:
        return f"{int(round(rate_khz))}K"
    return f"{rate_khz:g}K"


def build_export_tdms_name(start_time: datetime, sample_rate: float) -> str:
    return f"FIP-{format_sample_rate_token(sample_rate)}-{format_start_time_token(start_time)}.tdms"


def build_export_npz_name(start_time: datetime, sample_rate: float, sample_type: Optional[str] = None) -> str:
    sample_type_token = str(sample_type).strip().upper() if sample_type is not None and str(sample_type).strip() else "UNMARKED"
    return f"{sample_type_token}-FIP-{format_sample_rate_token(sample_rate)}-{format_start_time_token(start_time)}.npz"


def build_export_npz_suffix(start_time: datetime, sample_rate: float) -> str:
    return f"-FIP-{format_sample_rate_token(sample_rate)}-{format_start_time_token(start_time)}.npz"


def build_export_wav_name(start_time: datetime, sample_rate: float) -> str:
    return f"FIP-audio-{format_sample_rate_token(sample_rate)}-{format_start_time_token(start_time)}.wav"


def save_tdms_waveform(path: Path, phase_data: np.ndarray, sample_rate: float, start_time: datetime) -> Path:
    try:
        from nptdms import ChannelObject, RootObject, TdmsWriter
    except ImportError as exc:
        raise ImportError(
            "TDMS export requires the 'nptdms' package. Install it with 'pip install nptdms'."
        ) from exc

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(phase_data, dtype=np.float64).reshape(-1)
    root = RootObject(
        properties={
            'start_time': start_time.isoformat(timespec='milliseconds'),
            'sample_rate': float(sample_rate),
            'channel_name': 'phase_data',
        }
    )
    channel = ChannelObject(
        'FIP',
        'phase_data',
        values,
        properties={
            'start_time': start_time.isoformat(timespec='milliseconds'),
            'sample_rate': float(sample_rate),
            'unit_string': 'rad',
        },
    )
    with TdmsWriter(destination) as writer:
        writer.write_segment([root, channel])
    return destination


def save_npz_waveform(
    path: Path,
    phase_data: np.ndarray,
    sample_rate: float,
    start_time: datetime,
    arrival_time: Optional[datetime] = None,
    sample_type: Optional[str] = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(phase_data, dtype=np.float64).reshape(-1)
    start_time_token = format_start_time_token(start_time)
    arrival_time_token = format_arrival_time_token(arrival_time) if arrival_time is not None else None
    sample_type_token = str(sample_type).strip().upper() if sample_type is not None and str(sample_type).strip() else None
    np.savez(
        destination,
        phase_data=values,
        sample_rate=float(sample_rate),
        comm_count=int(values.size),
        npts=int(values.size),
        timestamp=float(start_time.timestamp()),
        starttime=start_time_token,
        arrival_time=arrival_time_token,
        type=sample_type_token,
        data_info={
            "type": "phase_data_export_visible_segment",
            "length": int(values.size),
            "npts": int(values.size),
            "duration_seconds": float(values.size) / max(float(sample_rate), 1.0),
            "save_time": datetime.now().isoformat(timespec="milliseconds"),
            "starttime": start_time_token,
            "arrival_time": arrival_time_token,
            "sample_type": sample_type_token,
        },
    )
    return destination


def save_wav_waveform(path: Path, phase_data: np.ndarray, sample_rate: float) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(phase_data, dtype=np.int16).reshape(-1)
    wavfile.write(destination, int(sample_rate), values)
    return destination


def parse_start_time_from_name(path: Path) -> datetime:
    match = TIME_TOKEN_RE.search(path.name)
    if not match:
        return datetime.fromtimestamp(path.stat().st_mtime)

    value = match.group("stamp")
    if "." in value:
        main, frac = value.split(".", 1)
        frac = (frac + "000000")[:6]
        value = f"{main}.{frac}"
        return datetime.strptime(value, "%Y%m%dT%H%M%S.%f")
    return datetime.strptime(value, "%Y%m%dT%H%M%S")


def parse_sample_rate_from_name(path: Path) -> float:
    match = SAMPLE_RATE_TOKEN_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot determine sample rate from file name: {path.name}")
    return float(match.group("rate")) * 1_000.0


def parse_arrival_time_token(value: object) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="ignore").strip()
    else:
        text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None

    match = ARRIVAL_TIME_TOKEN_RE.search(text)
    if match:
        token = match.group("stamp")
        if "." in token:
            main, frac = token.split(".", 1)
            frac = (frac + "000000")[:6]
            token = f"{main}.{frac}"
            return datetime.strptime(token, "%Y%m%d%H%M%S.%f")
        return datetime.strptime(token, "%Y%m%d%H%M%S")

    legacy_match = TIME_TOKEN_RE.search(text)
    if legacy_match:
        token = legacy_match.group("stamp")
        if "." in token:
            main, frac = token.split(".", 1)
            frac = (frac + "000000")[:6]
            token = f"{main}.{frac}"
            return datetime.strptime(token, "%Y%m%dT%H%M%S.%f")
        return datetime.strptime(token, "%Y%m%dT%H%M%S")

    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"Unsupported arrival_time format: {text}") from exc


def parse_optional_sample_type(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="ignore").strip()
    else:
        text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    return text.upper()


def list_data_files(directory: Path, sort_field: SortField, ascending: bool) -> list[FileRecord]:
    if not directory.exists() or not directory.is_dir():
        raise NotADirectoryError(f"Invalid directory: {directory}")

    files = [
        FileRecord(
            path=path,
            name=path.name,
            mtime=path.stat().st_mtime,
            size=path.stat().st_size,
        )
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]

    reverse = not ascending
    if sort_field == SortField.NAME:
        files.sort(key=lambda item: item.name.lower(), reverse=reverse)
    else:
        files.sort(key=lambda item: (item.mtime, item.name.lower()), reverse=reverse)
    return files


def paginate_files(records: list[FileRecord], page_index: int, page_size: int) -> PagedFiles:
    total_count = len(records)
    page_count = max(1, math.ceil(total_count / page_size)) if total_count else 1
    bounded_page = min(max(page_index, 0), page_count - 1)
    start = bounded_page * page_size
    stop = start + page_size
    return PagedFiles(
        items=records[start:stop],
        page_index=bounded_page,
        page_count=page_count,
        total_count=total_count,
    )


def _read_scalar(data: np.lib.npyio.NpzFile, key: str, cast_type):
    return cast_type(np.asarray(data[key]).item())


def _load_npz_waveform(path: Path) -> LoadedWaveform:
    data_info = None
    warnings: list[str] = []
    arrival_time: Optional[datetime] = None
    sample_type: Optional[str] = None

    with np.load(path, allow_pickle=True) as data:
        phase_data = np.asarray(data["phase_data"], dtype=np.float64).reshape(-1)
        sample_rate = _read_scalar(data, "sample_rate", float)
        comm_count = _read_scalar(data, "comm_count", int)
        timestamp = _read_scalar(data, "timestamp", float)

        if "data_info" in data.files:
            try:
                data_info_raw = data["data_info"]
                if isinstance(data_info_raw, np.ndarray) and data_info_raw.shape == ():
                    data_info = data_info_raw.item()
                elif isinstance(data_info_raw, np.ndarray):
                    data_info = data_info_raw.tolist()
                else:
                    data_info = data_info_raw
            except Exception as exc:
                warnings.append(f"Failed to read data_info: {exc}")

        if "arrival_time" in data.files:
            try:
                arrival_raw = np.asarray(data["arrival_time"])
                arrival_value = arrival_raw.item() if arrival_raw.shape == () else arrival_raw.tolist()
                arrival_time = parse_arrival_time_token(arrival_value)
            except Exception as exc:
                warnings.append(f"Failed to parse arrival_time: {exc}")
        elif isinstance(data_info, dict) and "arrival_time" in data_info:
            try:
                arrival_time = parse_arrival_time_token(data_info["arrival_time"])
            except Exception as exc:
                warnings.append(f"Failed to parse data_info.arrival_time: {exc}")

        if "type" in data.files:
            try:
                type_raw = np.asarray(data["type"])
                type_value = type_raw.item() if type_raw.shape == () else type_raw.tolist()
                sample_type = parse_optional_sample_type(type_value)
            except Exception as exc:
                warnings.append(f"Failed to parse type: {exc}")
        elif isinstance(data_info, dict) and "sample_type" in data_info:
            try:
                sample_type = parse_optional_sample_type(data_info["sample_type"])
            except Exception as exc:
                warnings.append(f"Failed to parse data_info.sample_type: {exc}")

    warning = "; ".join(warnings) if warnings else None

    return LoadedWaveform(
        path=path,
        phase_data=phase_data,
        sample_rate=sample_rate,
        comm_count=comm_count,
        timestamp=timestamp,
        start_time=parse_start_time_from_name(path),
        data_info=data_info,
        data_info_warning=warning,
        arrival_time=arrival_time,
        sample_type=sample_type,
    )


def _load_tdms_waveform(path: Path) -> LoadedWaveform:
    try:
        from nptdms import TdmsFile
    except ImportError as exc:
        raise ImportError(
            "TDMS support requires the 'nptdms' package. Install it with 'pip install nptdms'."
        ) from exc

    tdms_file = TdmsFile.read(path)
    selected_channel = None
    for group in tdms_file.groups():
        channels = group.channels()
        if channels:
            selected_channel = channels[0]
            break

    if selected_channel is None:
        raise ValueError(f"No readable channels found in TDMS file: {path.name}")

    phase_data = np.asarray(selected_channel[:], dtype=np.float64).reshape(-1)
    sample_rate = parse_sample_rate_from_name(path)
    start_time = parse_start_time_from_name(path)
    return LoadedWaveform(
        path=path,
        phase_data=phase_data,
        sample_rate=sample_rate,
        comm_count=int(phase_data.size),
        timestamp=start_time.timestamp(),
        start_time=start_time,
        data_info={
            "source_format": "tdms",
            "group_name": selected_channel.group_name,
            "channel_name": selected_channel.name,
        },
        data_info_warning=None,
        arrival_time=None,
        sample_type=None,
    )


def load_waveform(path: Path) -> LoadedWaveform:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return _load_npz_waveform(path)
    if suffix == ".tdms":
        return _load_tdms_waveform(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")
