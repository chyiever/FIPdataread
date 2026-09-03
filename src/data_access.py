from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.io import wavfile

from models import FileRecord, LoadedWaveform, PagedFiles, SortField


TIME_TOKEN_RE = re.compile(r"(?P<stamp>\d{8}T\d{6}(?:\.\d{1,6})?)")
COMPACT_TIME_TOKEN_RE = re.compile(r"(?P<stamp>\d{14}(?:\.\d{1,6})?)")
FILENAME_DATE_TOKEN_RE = re.compile(
    r"(?P<year>\d{4})[-_](?P<month>\d{1,2})[-_](?P<day>\d{1,2})"
    r"[-_](?P<hour>\d{1,2})[-_](?P<minute>\d{1,2})[-_](?P<second>\d{1,2})"
)
SAMPLE_RATE_TOKEN_RE = re.compile(
    r"(?:^|[-_])(?P<rate>\d+(?:\.\d+)?)(?P<unit>mhz|m|khz|k)(?=$|[-_])",
    re.IGNORECASE,
)
ARRIVAL_TIME_TOKEN_RE = re.compile(r"(?P<stamp>\d{14}(?:\.\d{1,6})?)")
SUPPORTED_SUFFIXES = {".npz", ".tdms", ".txt"}


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


def build_export_npz_name(start_time: datetime, sample_rate: float) -> str:
    return f"FIP-{format_sample_rate_token(sample_rate)}-{format_start_time_token(start_time)}.npz"


def build_export_txt_name(start_time: datetime, sample_rate: float) -> str:
    return f"FIP-{format_sample_rate_token(sample_rate)}-{format_start_time_token(start_time)}.txt"


def build_export_wav_name(start_time: datetime, sample_rate: float) -> str:
    return f"FIP-audio-{format_sample_rate_token(sample_rate)}-{format_start_time_token(start_time)}.wav"


def _normalize_channel_export_data(phase_data: np.ndarray | tuple[np.ndarray, ...] | list[np.ndarray]) -> tuple[np.ndarray, ...]:
    if isinstance(phase_data, (tuple, list)):
        channels = tuple(np.asarray(channel, dtype=np.float64).reshape(-1) for channel in phase_data)
    else:
        values = np.asarray(phase_data, dtype=np.float64)
        if values.ndim == 2:
            channels = tuple(values[:, index].reshape(-1) for index in range(values.shape[1]))
        else:
            channels = (values.reshape(-1),)

    if not channels:
        raise ValueError("At least one channel is required for export.")

    sample_count = channels[0].size
    if any(channel.size != sample_count for channel in channels):
        raise ValueError("All exported channels must have the same sample count.")
    return channels


def _channels_to_export_array(channels: tuple[np.ndarray, ...]) -> np.ndarray:
    if len(channels) == 1:
        return channels[0]
    return np.column_stack(channels)


def save_tdms_waveform(path: Path, phase_data: np.ndarray | tuple[np.ndarray, ...] | list[np.ndarray], sample_rate: float, start_time: datetime) -> Path:
    try:
        from nptdms import ChannelObject, RootObject, TdmsWriter
    except ImportError as exc:
        raise ImportError(
            "TDMS export requires the 'nptdms' package. Install it with 'pip install nptdms'."
        ) from exc

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    channels = _normalize_channel_export_data(phase_data)
    channel_names = tuple("phase_data" if index == 0 else f"phase_data_ch{index + 1}" for index in range(len(channels)))
    root = RootObject(
        properties={
            'start_time': start_time.isoformat(timespec='milliseconds'),
            'sample_rate': float(sample_rate),
            'channel_name': channel_names[0],
            'channel_count': len(channels),
            'channel_names': ",".join(channel_names),
        }
    )
    channel_objects = (
        ChannelObject(
            'FIP',
            channel_name,
            values,
            properties={
                'start_time': start_time.isoformat(timespec='milliseconds'),
                'sample_rate': float(sample_rate),
                'unit_string': 'rad',
            },
        )
        for channel_name, values in zip(channel_names, channels)
    )
    with TdmsWriter(destination) as writer:
        writer.write_segment([root, *channel_objects])
    return destination


def save_npz_waveform(
    path: Path,
    phase_data: np.ndarray | tuple[np.ndarray, ...] | list[np.ndarray],
    sample_rate: float,
    start_time: datetime,
    arrival_time: Optional[datetime] = None,
    sample_type: Optional[str] = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    channels = _normalize_channel_export_data(phase_data)
    values = _channels_to_export_array(channels)
    channel_names = tuple("phase_data" if index == 0 else f"phase_data_ch{index + 1}" for index in range(len(channels)))
    sample_count = int(channels[0].size)
    start_time_token = format_start_time_token(start_time)
    arrival_time_token = format_arrival_time_token(arrival_time) if arrival_time is not None else None
    sample_type_token = str(sample_type).strip().upper() if sample_type is not None and str(sample_type).strip() else None
    np.savez(
        destination,
        phase_data=values,
        channels=np.column_stack(channels),
        channel_names=np.asarray(channel_names),
        channel_count=len(channels),
        sample_rate=float(sample_rate),
        comm_count=sample_count,
        npts=sample_count,
        timestamp=float(start_time.timestamp()),
        starttime=start_time_token,
        arrival_time=arrival_time_token,
        type=sample_type_token,
        data_info={
            "type": "phase_data_export_visible_segment",
            "length": sample_count,
            "npts": sample_count,
            "duration_seconds": float(sample_count) / max(float(sample_rate), 1.0),
            "save_time": datetime.now().isoformat(timespec="milliseconds"),
            "starttime": start_time_token,
            "arrival_time": arrival_time_token,
            "sample_type": sample_type_token,
            "channel_count": len(channels),
            "channel_names": channel_names,
        },
    )
    return destination


def save_txt_waveform(path: Path, phase_data: np.ndarray | tuple[np.ndarray, ...] | list[np.ndarray]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    channels = _normalize_channel_export_data(phase_data)
    np.savetxt(destination, _channels_to_export_array(channels), fmt="%.18e")
    return destination


def save_wav_waveform(path: Path, phase_data: np.ndarray, sample_rate: float) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(phase_data, dtype=np.int16).reshape(-1)
    wavfile.write(destination, int(sample_rate), values)
    return destination


def parse_start_time_from_name(path: Path) -> datetime:
    match = TIME_TOKEN_RE.search(path.name)
    if match:
        value = match.group("stamp")
        if "." in value:
            main, frac = value.split(".", 1)
            frac = (frac + "000000")[:6]
            value = f"{main}.{frac}"
            return datetime.strptime(value, "%Y%m%dT%H%M%S.%f")
        return datetime.strptime(value, "%Y%m%dT%H%M%S")

    compact_match = COMPACT_TIME_TOKEN_RE.search(path.name)
    if compact_match:
        value = compact_match.group("stamp")
        if "." in value:
            main, frac = value.split(".", 1)
            frac = (frac + "000000")[:6]
            value = f"{main}.{frac}"
            return datetime.strptime(value, "%Y%m%d%H%M%S.%f")
        return datetime.strptime(value, "%Y%m%d%H%M%S")

    date_match = FILENAME_DATE_TOKEN_RE.search(path.name)
    if date_match:
        return datetime(
            year=int(date_match.group("year")),
            month=int(date_match.group("month")),
            day=int(date_match.group("day")),
            hour=int(date_match.group("hour")),
            minute=int(date_match.group("minute")),
            second=int(date_match.group("second")),
        )

    return datetime.fromtimestamp(path.stat().st_mtime)


def parse_sample_rate_from_name(path: Path) -> float:
    match = SAMPLE_RATE_TOKEN_RE.search(path.stem)
    if not match:
        raise ValueError(f"Cannot determine sample rate from file name: {path.name}")
    unit = match.group("unit").lower()
    factor = 1_000_000.0 if unit.startswith("m") else 1_000.0
    return float(match.group("rate")) * factor


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
        raw_phase_data = np.asarray(data["phase_data"], dtype=np.float64)
        if raw_phase_data.ndim == 2:
            if raw_phase_data.shape[1] < 1:
                raise ValueError(f"NPZ phase_data contains no channels: {path.name}")
            channel_arrays = tuple(raw_phase_data[:, index].reshape(-1) for index in range(raw_phase_data.shape[1]))
        elif "channels" in data.files:
            raw_channels = np.asarray(data["channels"], dtype=np.float64)
            if raw_channels.ndim == 2:
                channel_arrays = tuple(raw_channels[:, index].reshape(-1) for index in range(raw_channels.shape[1]))
            else:
                channel_arrays = (raw_channels.reshape(-1),)
        else:
            channel_arrays = (raw_phase_data.reshape(-1),)
        phase_data = channel_arrays[0]
        sample_rate = _read_scalar(data, "sample_rate", float)
        comm_count = _read_scalar(data, "comm_count", int)
        timestamp = _read_scalar(data, "timestamp", float)
        channel_names = tuple(f"CH{index + 1}" for index in range(len(channel_arrays)))
        if "channel_names" in data.files:
            try:
                loaded_names = tuple(str(item) for item in np.asarray(data["channel_names"]).reshape(-1).tolist())
                if loaded_names:
                    channel_names = loaded_names
            except Exception as exc:
                warnings.append(f"Failed to read channel_names: {exc}")

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
        channels=channel_arrays,
        channel_names=channel_names,
    )


def _load_tdms_waveform(path: Path) -> LoadedWaveform:
    try:
        from nptdms import TdmsFile
    except ImportError as exc:
        raise ImportError(
            "TDMS support requires the 'nptdms' package. Install it with 'pip install nptdms'."
        ) from exc

    tdms_file = TdmsFile.read(path)
    selected_group = None
    selected_channels = []
    for group in tdms_file.groups():
        channels = group.channels()
        if channels:
            selected_group = group
            selected_channels = channels
            break

    if not selected_channels or selected_group is None:
        raise ValueError(f"No readable channels found in TDMS file: {path.name}")

    channel_arrays = tuple(
        np.asarray(channel[:], dtype=np.float64).reshape(-1)
        for channel in selected_channels
    )
    channel_names = tuple(
        str(channel.name).strip() or f"CH{index + 1}"
        for index, channel in enumerate(selected_channels)
    )
    phase_data = channel_arrays[0]
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
            "group_name": selected_group.name,
            "channel_name": channel_names[0],
            "channel_names": channel_names,
            "channel_count": len(channel_arrays),
        },
        data_info_warning=None,
        arrival_time=None,
        sample_type=None,
        channels=channel_arrays,
        channel_names=channel_names,
    )


def _read_txt_array(path: Path) -> np.ndarray:
    errors: list[str] = []
    for delimiter in (None, ","):
        try:
            return np.loadtxt(path, dtype=np.float64, comments="#", delimiter=delimiter, ndmin=2)
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError(f"Cannot read TXT numeric columns from {path.name}: {'; '.join(errors)}")


def _load_txt_waveform(path: Path) -> LoadedWaveform:
    values = np.asarray(_read_txt_array(path), dtype=np.float64)
    if values.size == 0:
        raise ValueError(f"TXT file contains no numeric samples: {path.name}")

    if values.ndim == 1:
        channel_arrays = (values.reshape(-1),)
    elif values.ndim == 2:
        if values.shape[1] not in (1, 2):
            raise ValueError(
                f"TXT file must contain one or two numeric columns, got {values.shape[1]} columns: {path.name}"
            )
        channel_arrays = tuple(values[:, index].reshape(-1) for index in range(values.shape[1]))
    else:
        raise ValueError(f"TXT file must be a one- or two-column numeric table: {path.name}")

    phase_data = channel_arrays[0]
    sample_rate = parse_sample_rate_from_name(path)
    start_time = parse_start_time_from_name(path)
    channel_names = tuple(f"txt_column_{index + 1}" for index in range(len(channel_arrays)))
    return LoadedWaveform(
        path=path,
        phase_data=phase_data,
        sample_rate=sample_rate,
        comm_count=int(phase_data.size),
        timestamp=start_time.timestamp(),
        start_time=start_time,
        data_info={
            "source_format": "txt",
            "column_count": len(channel_arrays),
            "channel_names": channel_names,
            "filename_sample_rate": sample_rate,
            "filename_start_time": start_time.isoformat(timespec="milliseconds"),
        },
        data_info_warning=None,
        arrival_time=None,
        sample_type=None,
        channels=channel_arrays,
        channel_names=channel_names,
    )


def load_waveform(path: Path) -> LoadedWaveform:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return _load_npz_waveform(path)
    if suffix == ".tdms":
        return _load_tdms_waveform(path)
    if suffix == ".txt":
        return _load_txt_waveform(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def load_waveforms_concatenated(paths: Sequence[Path]) -> LoadedWaveform:
    selected_paths = [Path(path) for path in paths]
    if not selected_paths:
        raise ValueError("At least one file is required.")
    if len(selected_paths) == 1:
        return load_waveform(selected_paths[0])

    waveforms = sorted((load_waveform(path) for path in selected_paths), key=lambda item: item.start_time)
    sample_rate = float(waveforms[0].sample_rate)
    channel_count = waveforms[0].channel_count
    channel_names = waveforms[0].channel_names
    for waveform in waveforms[1:]:
        if abs(float(waveform.sample_rate) - sample_rate) > max(1e-9, sample_rate * 1e-9):
            raise ValueError("Selected files must have the same sample rate before concatenation.")
        if waveform.channel_count != channel_count:
            raise ValueError("Selected files must have the same channel count before concatenation.")

    concatenated_channels = tuple(
        np.concatenate([waveform.channel_data(channel_index) for waveform in waveforms])
        for channel_index in range(channel_count)
    )
    phase_data = concatenated_channels[0]
    start_time = waveforms[0].start_time
    source_files = [str(waveform.path) for waveform in waveforms]
    display_name = f"{waveforms[0].path.stem}+{len(waveforms)}files{waveforms[0].path.suffix}"

    return LoadedWaveform(
        path=waveforms[0].path.with_name(display_name),
        phase_data=phase_data,
        sample_rate=sample_rate,
        comm_count=int(phase_data.size),
        timestamp=start_time.timestamp(),
        start_time=start_time,
        data_info={
            "source_format": "concatenated",
            "source_files": source_files,
            "file_count": len(waveforms),
            "channel_count": channel_count,
            "channel_names": channel_names,
            "start_time": start_time.isoformat(timespec="milliseconds"),
            "end_time": (
                start_time + timedelta(seconds=float(phase_data.size) / max(sample_rate, 1.0))
            ).isoformat(timespec="milliseconds"),
        },
        data_info_warning=None,
        arrival_time=None,
        sample_type=None,
        channels=concatenated_channels,
        channel_names=channel_names,
    )
