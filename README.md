# FIPread

FIPread is a desktop tool for browsing and analyzing FIP waveform files (`.npz`, `.tdms`).

## Features

- File list with sorting, paging, and threshold-based filtering
- Time-domain waveform display with optional filter
- TDMS input supports one or two channels; the top time plot uses channel 1 by default
- Plot 2 mode switch: `SVM Prediction` or `Short-Time Energy`
- Plot 2 can also display the second TDMS channel waveform with the same display filter preprocessing
- PSD analysis from a selected visible window, with channel 1, channel 2, or both-channel display options
- Visible waveform audio playback (`Play`, `Stop`, `Replay`)
- Visible waveform export to `.wav`
- Visible raw waveform export

## Environment

Recommended:

- Windows
- Python 3.9+

Install dependencies (batch):

```powershell
pip install -r requirements.txt
```

The full library list and version details are documented in `docs/required-libraries.md`.

## Run

From project root:

```powershell
python .\run.py
```

## Build EXE

Install packaging dependencies (batch):

```powershell
pip install -r requirements-build.txt
```

From project root:

```powershell
python .\build_exe.py
```

Default output:

```text
dist\FIP.YYYY.MM.DD.exe
```

Example:

```text
dist\FIP.2026.08.01.exe
```

Packaging behavior:

- Existing exe files in `dist` are preserved by default.
- If `dist\FIP.YYYY.MM.DD.exe` already exists, the new exe is saved with a time suffix, for example `FIP.2026.08.01.203012.exe`.
- PyInstaller intermediate files are staged under `build` and removed after a successful build.
- Runtime resources `logo.png` and `models/saved_models` are bundled into the exe.
- Matplotlib colormaps are included so the `t-f Plot` color bar matches source-mode display.

Useful variants:

```powershell
python .\build_exe.py --console
python .\build_exe.py --clean-only
python .\build_exe.py --keep-intermediate
python .\build_exe.py --overwrite
python .\build_exe.py --collect-sklearn
```

## Basic Workflow

1. Choose a data directory.
2. Select a file from the list.
3. View waveform in the top plot.
4. Adjust filter parameters in `Display Controls` if needed.
5. Use `Zoom Mode` or `Window PSD Mode` for interaction.
6. Use `Apply Visible Window` to set visible duration.
7. For two-channel TDMS files, choose `Channel 2 Waveform` in `Plot 2` to compare the second channel after the same display preprocessing.
8. For PSD, use the `PSD` dropdown to choose `Channel 1`, `Channel 2`, or `Both Channels`; the default is always `Channel 1`.
9. In `Plot 2`, choose `SVM Prediction` or `Short-Time Energy` for the existing first-channel feature workflows.
10. Use audio controls to listen/export the current visible segment.

## Notes

- X-axis time labels are derived from filename timestamp + sample rate.
- Legacy TDMS sample-rate tokens such as `-200K-` are supported.
- New TDMS sample-rate tokens such as `-1MHz-` are supported.
- Legacy start-time tokens such as `20260323T103125.631` are supported.
- New TDMS start-time tokens such as `2026-8-1-12-43-36` are supported.
- The top time plot, visible raw export, audio playback/export, SVM prediction, short-time energy, and t-f plot use channel 1.
- Plot 2 `Channel 2 Waveform` applies the same display filter settings as the top time plot.
- PSD uses the selected raw window and can draw channel 1, channel 2, or both channels.
