# FIPread

FIPread is a desktop tool for browsing and analyzing FIP waveform files (`.npz`, `.tdms`, `.txt`).

## Features

- File list with sorting, paging, and threshold-based filtering
- Time-domain waveform display with optional filter
- TDMS input supports one or two channels; the top time plot uses CH1 by default
- TXT input supports one or two numeric columns; each column is treated as one channel
- Plot 2 mode switch: `SVM Prediction` or `Short-Time Energy`
- Plot 2 can also display the second TDMS channel waveform with the same display filter preprocessing
- PSD analysis from a selected visible window, with CH1, CH2, or CH1+CH2 display options
- PSD X/Y display ranges can be adjusted from `Display Controls`
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
2. Select a file from the list, or use Ctrl/extended selection to select multiple compatible files.
3. View waveform in the top plot. Multi-selected files are sorted by start time and concatenated end-to-start before plotting.
4. Adjust filter parameters in `Display Controls` if needed.
5. Use `Zoom Mode` or `Window PSD Mode` for interaction.
6. Use `Apply Visible Window` to set visible duration.
7. For two-channel TDMS files, choose `CH2 Waveform` in `Plot 2` to compare the second channel after the same display preprocessing.
8. For PSD, use the `PSD` dropdown to choose `CH1`, `CH2`, or `CH1+CH2`; the default is always `CH1`.
9. For t-f analysis, open the `t-f Plot` tab and use its compact `CH` dropdown to choose `CH1` or `CH2`; the selected channel drives the single time plot in that tab and the t-f plot below it.
10. In `Plot 2`, choose `SVM Prediction` or `Short-Time Energy` for the existing first-channel feature workflows.
11. Use audio controls to listen/export the current visible segment.

## Notes

- X-axis time labels are derived from filename timestamp + sample rate.
- Legacy TDMS/TXT sample-rate tokens such as `-200K-` are supported.
- New TDMS/TXT sample-rate tokens such as `-1MHz-` and `_1000k` are supported.
- Legacy start-time tokens such as `20260323T103125.631` are supported.
- New TDMS start-time tokens such as `2026-8-1-12-43-36` are supported.
- Compact TXT start-time tokens such as `20260820170538.780` are supported.
- Visible raw export preserves every loaded channel; the top time plot, audio playback/export, SVM prediction, and short-time energy use CH1 by default.
- Ctrl/extended multi-selection in the file list concatenates selected files by parsed start time. Selected files must have the same sample rate and channel count.
- The t-f plot defaults to CH1 and can switch to CH2 from the `t-f Plot` tab when the loaded file has two channels.
- Plot 2 `CH2 Waveform` applies the same display filter settings as the top time plot.
- PSD uses the selected raw window and can draw CH1, CH2, or CH1+CH2.
- When the PSD X range spans at least one decade, the X-axis labels are limited to powers of ten, such as `100Hz`, `1000Hz`, and `10000Hz`.
