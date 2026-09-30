"""Core data types shared across the application.

Contains the file-list record, the loaded-waveform container and the small
enumerations that select sort order, filter shape and plot interaction mode.
"""

from __future__ import annotations


from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np


PAGE_SIZE = 100


class SortField(str, Enum):
    """File-list sort keys (file name or modification time).
    """

    NAME = "name"
    MTIME = "mtime"


class FilterMode(str, Enum):
    """Display-filter response shapes available to the user.
    """

    BANDPASS = "Band-pass"
    HIGHPASS = "High-pass"
    LOWPASS = "Low-pass"


class InteractionMode(str, Enum):
    """Mouse behaviour of a time-domain plot.
    """

    ZOOM = "Zoom"
    WINDOW_PSD = "Window PSD"


@dataclass(frozen=True)
class FileRecord:
    """A single file-list entry (path, name, mtime, size).
    """

    path: Path
    name: str
    mtime: float
    size: int


@dataclass
class LoadedWaveform:
    """An in-memory waveform plus all metadata parsed from disk.

    ``phase_data`` always aliases channel 0 so that existing single-channel
    code paths keep working, while ``channels`` holds every channel found
    in the file.
    """

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
        """Back-fill ``channels``/``channel_names`` from ``phase_data``.

        Guarantees ``channels`` has at least one entry and that ``channel_names`` is at
        least as long as ``channels``, auto-naming any surplus channels ``CH3``, ``CH4``...
        """

        if not self.channels:
            self.channels = (self.phase_data,)
        if not self.channel_names:
            self.channel_names = tuple(f"CH{index + 1}" for index in range(len(self.channels)))
        elif len(self.channel_names) < len(self.channels):
            names = list(self.channel_names)
            names.extend(f"CH{index + 1}" for index in range(len(names), len(self.channels)))
            self.channel_names = tuple(names)

    @property
    def channel_count(self) -> int:
        """Number of waveform channels carried by this record.
        """

        return len(self.channels)

    def channel_data(self, channel_index: int) -> np.ndarray:
        """Return the array for ``channel_index``.

        Raises:
            IndexError: if ``channel_index`` is outside the available channels.
        """

        if 0 <= channel_index < len(self.channels):
            return self.channels[channel_index]
        raise IndexError(f"Channel index out of range: {channel_index}")

    def channel_label(self, channel_index: int) -> str:
        """Return a display label such as ``CH1`` or ``CH1: Outer``.

        Falls back to the plain ``CH<n>`` label when the stored name is empty or
        identical to the default.
        """

        base_label = f"CH{channel_index + 1}"
        if 0 <= channel_index < len(self.channel_names):
            name = str(self.channel_names[channel_index]).strip()
            if name and name != base_label:
                return f"{base_label}: {name}"
        return base_label


@dataclass(frozen=True)
class PagedFiles:
    """One page of file-list results plus paging state.
    """

    items: list[FileRecord]
    page_index: int
    page_count: int
    total_count: int
