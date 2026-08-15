from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np


PAGE_SIZE = 100


class SortField(str, Enum):
    NAME = "name"
    MTIME = "mtime"


class FilterMode(str, Enum):
    BANDPASS = "Band-pass"
    HIGHPASS = "High-pass"
    LOWPASS = "Low-pass"


class InteractionMode(str, Enum):
    ZOOM = "Zoom"
    WINDOW_PSD = "Window PSD"


@dataclass(frozen=True)
class FileRecord:
    path: Path
    name: str
    mtime: float
    size: int


@dataclass
class LoadedWaveform:
    path: Path
    phase_data: np.ndarray
    sample_rate: float
    comm_count: int
    timestamp: float
    start_time: datetime
    data_info: Optional[Any]
    data_info_warning: Optional[str] = None
    arrival_time: Optional[datetime] = None
    sample_type: Optional[str] = None
    channels: tuple[np.ndarray, ...] = ()
    channel_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.channels:
            self.channels = (self.phase_data,)
        if not self.channel_names:
            self.channel_names = tuple(f"Channel {index + 1}" for index in range(len(self.channels)))
        elif len(self.channel_names) < len(self.channels):
            names = list(self.channel_names)
            names.extend(f"Channel {index + 1}" for index in range(len(names), len(self.channels)))
            self.channel_names = tuple(names)

    @property
    def channel_count(self) -> int:
        return len(self.channels)

    def channel_data(self, channel_index: int) -> np.ndarray:
        if 0 <= channel_index < len(self.channels):
            return self.channels[channel_index]
        raise IndexError(f"Channel index out of range: {channel_index}")

    def channel_label(self, channel_index: int) -> str:
        base_label = f"Channel {channel_index + 1}"
        if 0 <= channel_index < len(self.channel_names):
            name = str(self.channel_names[channel_index]).strip()
            if name and name != base_label:
                return f"{base_label}: {name}"
        return base_label


@dataclass(frozen=True)
class PagedFiles:
    items: list[FileRecord]
    page_index: int
    page_count: int
    total_count: int
