from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .models import FileRecord, LoadedWaveform, PagedFiles, SortField


TIME_TOKEN_RE = re.compile(r"(?P<stamp>\d{8}T\d{6}(?:\.\d{1,6})?)")
SAMPLE_RATE_TOKEN_RE = re.compile(r"-(?P<rate>\d+(?:\.\d+)?)K-", re.IGNORECASE)
SUPPORTED_SUFFIXES = {".npz", ".tdms"}


def format_start_time_token(start_time: datetime) -> str:
    return start_time.strftime("%Y%m%dT%H%M%S.%f")[:-3]


def format_sample_rate_token(sample_rate: float) -> str:
    rate_khz = float(sample_rate) / 1_000.0
    if abs(rate_khz - round(rate_khz)) < 1e-9:
        return f"{int(round(rate_khz))}K"
    return f"{rate_khz:g}K"


def build_export_tdms_name(start_time: datetime, sample_rate: float) -> str:
    return f"FIP-{format_sample_rate_token(sample_rate)}-{format_start_time_token(start_time)}.tdms"


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
    warning: Optional[str] = None

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
                warning = f"Failed to read data_info: {exc}"

    return LoadedWaveform(
        path=path,
        phase_data=phase_data,
        sample_rate=sample_rate,
        comm_count=comm_count,
        timestamp=timestamp,
        start_time=parse_start_time_from_name(path),
        data_info=data_info,
        data_info_warning=warning,
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
    )


def load_waveform(path: Path) -> LoadedWaveform:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return _load_npz_waveform(path)
    if suffix == ".tdms":
        return _load_tdms_waveform(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")
