"""Application-wide constant identifiers shared across the FIPread UI modules.

These values are plain string tags rather than an ``Enum`` because they are used
directly as widget item data and as dictionary keys in the main window, and they
are compared against raw text coming back from Qt combo boxes.

Keeping them in a dedicated module (instead of leaving them in
``main_window``) lets every mixin module import them without creating a circular
import between the mixins and the composed ``MainWindow`` class.
"""

from __future__ import annotations

# --- Plot 2 feature modes -------------------------------------------------
# Value stored in the Plot 2 combo box and used to dispatch feature rendering.
FEATURE_MODE_NONE = "none"
FEATURE_MODE_CHANNEL_1 = "channel_1_waveform"
FEATURE_MODE_CHANNEL_2 = "channel_2_waveform"
FEATURE_MODE_SVM = "svm_prediction"
FEATURE_MODE_ENERGY = "short_time_energy"
FEATURE_MODE_BAND_ENERGY = "short_time_band_energy"
FEATURE_MODE_ENERGY_ENERGY = "short_time_energy_energy"
FEATURE_MODE_PSD_SUM = "short_time_psd_sum"
FEATURE_MODE_MAX_NUM = "st_energy_max_num"

# --- PSD channel source ---------------------------------------------------
# Value stored in the PSD combo box; selects which channel(s) feed the Welch PSD.
PSD_SOURCE_CHANNEL_1 = "channel_1"
PSD_SOURCE_CHANNEL_2 = "channel_2"
PSD_SOURCE_BOTH = "both"

# --- Time-frequency channel source ----------------------------------------
# Value stored in the compact CH combo box on the t-f Plot tab.
TF_SOURCE_CHANNEL_1 = "channel_1"
TF_SOURCE_CHANNEL_2 = "channel_2"

# --- Time-frequency spectrogram mode --------------------------------------
# ``psd`` -> scipy.signal.spectrogram(scaling="density");
# ``amplitude`` -> scipy.signal.spectrogram(scaling="spectrum").
TF_MODE_PSD = "psd"
TF_MODE_AMPLITUDE = "amplitude"

# --- Time-frequency value scale ------------------------------------------
TF_SCALE_LOG = "log"
TF_SCALE_LINEAR = "linear"
