from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FilePanelDefaults:
    input_directory: str = "data"
    export_directory: str = "exports"


@dataclass(frozen=True)
class DisplayPanelDefaults:
    filter_enabled: bool = True
    filter_mode_index: int = 1
    low_cut_hz: float = 10000.0
    high_cut_hz: float = 50000.0
    phase_y_min: float = 0.0
    phase_y_max: float = 0.0
    tf_window_seconds: float = 0.0008
    tf_overlap_percent: float = 80.0
    tf_y_min_hz: float = 0.0
    tf_y_max_hz: float = 0.0
    tf_colormap: str = "jet"
    tf_color_auto: bool = True
    tf_color_min: float = -120.0
    tf_color_max: float = 0.0


@dataclass(frozen=True)
class SamplePanelDefaults:
    sample_length_seconds: float = 0.02
    hop_seconds: float = 0.005
    default_codes: tuple[str, ...] = ("F130", "F130A", "F130B", "F130C", "OT")


@dataclass(frozen=True)
class UIPanelDefaults:
    file: FilePanelDefaults = field(default_factory=FilePanelDefaults)
    display: DisplayPanelDefaults = field(default_factory=DisplayPanelDefaults)
    sample: SamplePanelDefaults = field(default_factory=SamplePanelDefaults)


UI_DEFAULTS = UIPanelDefaults()
