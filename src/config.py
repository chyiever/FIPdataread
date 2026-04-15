from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FilePanelDefaults:
    amplitude_threshold: float = 0.01


@dataclass(frozen=True)
class DisplayPanelDefaults:
    filter_enabled: bool = True
    filter_mode_index: int = 1
    low_cut_hz: float = 10000.0
    high_cut_hz: float = 50000.0
    phase_y_min: float = 0.0
    phase_y_max: float = 0.0
    psd_y_min: int = -120
    psd_y_max: int = -45

    tf_mode_index: int = 0
    tf_value_scale_index: int = 0
    tf_window_seconds: float = 0.0008
    tf_overlap_percent: float = 80.0
    tf_y_min_hz: float = 0.0
    tf_y_max_hz: float = 0.0
    tf_colormap: str = "jet"
    tf_color_auto: bool = True
    tf_color_min: float = -120.0
    tf_color_max: float = 0.0


@dataclass(frozen=True)
class FeaturePanelDefaults:
    band1_low_hz: float = 4000.0
    band1_high_hz: float = 10000.0
    band2_low_hz: float = 20000.0
    band2_high_hz: float = 40000.0
    window_seconds: float = 0.03
    step_percent: float = 50.0
    amplitude_gate: float = 0.02
    y_min: float = 0.0
    y_max: float = 0.0


@dataclass(frozen=True)
class AudioPanelDefaults:
    downsample_factor: int = 10


@dataclass(frozen=True)
class ViewDefaults:
    zoom_mode_checked: bool = True
    visible_window_seconds: float = 1.0


@dataclass(frozen=True)
class UIPanelDefaults:
    file: FilePanelDefaults = field(default_factory=FilePanelDefaults)
    display: DisplayPanelDefaults = field(default_factory=DisplayPanelDefaults)
    feature: FeaturePanelDefaults = field(default_factory=FeaturePanelDefaults)
    audio: AudioPanelDefaults = field(default_factory=AudioPanelDefaults)
    view: ViewDefaults = field(default_factory=ViewDefaults)


UI_DEFAULTS = UIPanelDefaults()
