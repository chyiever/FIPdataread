# FIPread

FIPread is a desktop tool for browsing and analyzing FIP waveform files (`.npz`, `.tdms`).

## Features

- File list with sorting, paging, and threshold-based filtering
- Time-domain waveform display with optional filter
- Plot 2 mode switch: `SVM Prediction` or `Short-Time Energy`
- PSD analysis from a selected visible window
- Visible waveform audio playback (`Play`, `Stop`, `Replay`)
- Visible waveform export to `.wav`
- Visible raw waveform export

## Environment

Recommended:

- Windows
- Python 3.9+

Install dependencies:

```powershell
pip install numpy scipy PyQt5 pyqtgraph nptdms pandas joblib scikit-learn
```

## Run

From project root:

```powershell
python .\run.py
```

## Basic Workflow

1. Choose a data directory.
2. Select a file from the list.
3. View waveform in the top plot.
4. Adjust filter parameters in `Display Controls` if needed.
5. Use `Zoom Mode` or `Window PSD Mode` for interaction.
6. Use `Apply Visible Window` to set visible duration.
7. In `Plot 2`, choose `SVM Prediction` or `Short-Time Energy`.
8. Use audio controls to listen/export the current visible segment.

## Notes

- X-axis time labels are derived from filename timestamp + sample rate.
- PSD always uses the selected raw window.
- Audio playback/export uses the currently visible display waveform.
