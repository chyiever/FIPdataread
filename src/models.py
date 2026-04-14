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


@dataclass(frozen=True)
class PagedFiles:
    items: list[FileRecord]
    page_index: int
    page_count: int
    total_count: int
