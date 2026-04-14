# FIPread Requirement Specification v1.0

## 1. Document Information

- Project name: FIPread
- Version: v1.0
- Document type: Development requirement specification
- Target platform: Windows desktop
- Development language: Python 3.9+
- UI framework: PyQt5
- Main plotting library: pyqtgraph
- Numerical libraries: NumPy, SciPy
- TDMS reader: nptdms

## 2. Project Overview

FIPread is a desktop application for loading, browsing, plotting, and analyzing FIP phase data stored in `.npz` and `.tdms` files. The application targets high-sample-rate waveform data and must provide efficient browsing, time-domain visualization, window-based PSD analysis, and lightweight filtering for display.

## 3. Data Source and File Protocol

### 3.1 Supported Input

- Input type: directory only
- Recursive scan: not supported
- Displayed file type: `.npz` and `.tdms`

### 3.2 Data Protocol

For `.npz`, the actual structure shall follow the sample files in the `data` directory. The application shall at least support these top-level fields:

- `phase_data`
- `sample_rate`
- `comm_count`
- `timestamp`
- `data_info`

For `.tdms`, the application shall read the waveform from the first available channel, use the file name timestamp as the waveform start time, and parse the sample rate from file-name tokens such as `500K` or `1000K`.

### 3.3 Field Handling Rules

- `phase_data` is the primary waveform data and is mandatory.
- `sample_rate` is mandatory and shall be used for time-axis and PSD calculation.
- `comm_count` and `timestamp` are metadata fields and should be loaded if present.
- `data_info` is optional metadata.
- If `data_info` cannot be deserialized because of environment or NumPy compatibility problems, the software shall degrade gracefully and continue to open the file.
- If required core fields are missing or unreadable, the software shall reject the file and report an error.

### 3.4 Start Time Rule

- File start absolute time shall be parsed from the file name.
- The file naming pattern is expected to contain a time token like `YYYYMMDDTHHMMSS.mmm`.
- If time parsing fails, the software shall fall back to file modification time and show a warning.

## 4. Functional Requirements

### 4.1 Directory Input and File List

The software shall provide:

- A directory input box
- A browse button for selecting a directory
- A file list showing only `.npz` files in the selected directory

The file list shall support:

- Pagination by 30 files per page
- `Home`
- `Previous 30`
- `Next 30`
- `End`
- Current page number
- Total page count
- Total file count

The file list shall support sorting by:

- File name
- File modification time

Sorting shall support:

- Ascending
- Descending

### 4.2 File Selection and Time-Domain Plot

When the user selects a file from the list:

- The software shall load the waveform data.
- The upper plot area shall display the time-domain waveform.
- The horizontal axis shall represent absolute time derived from the file name.
- The vertical axis shall represent phase in `rad`.

### 4.3 Y-Axis Range Control

The UI shall provide two floating-point input boxes for Y-axis minimum and maximum.

Rules:

- If both values are `0`, the Y-axis shall use automatic scaling.
- Otherwise, the software shall use the input values as manual limits.
- If the input is invalid, the software shall fall back to automatic scaling.

### 4.4 Plot Interaction Modes

The software shall provide two exclusive interaction modes:

- `Zoom` mode
- `Window PSD` mode

#### Zoom Mode

The software shall support:

- Rectangle zoom in
- Zoom out
- Reset to full range

#### Window PSD Mode

The software shall support:

- Left-drag selection of a short-time window
- Both left-to-right and right-to-left drag directions
- Right-click clear current selection
- Only one selection window at a time
- A new selection shall overwrite the previous one
- Selection beyond signal boundaries shall not continue
- Selected region shall be shown as yellow shading on the time-domain plot

Window selection rule:

- Minimum window length: `0.001 s`

### 4.5 PSD Analysis

The lower plot area shall display the PSD of the selected time-domain window.

Requirements:

- Plot layout is fixed: upper plot is time-domain, lower plot is PSD
- PSD shall be calculated from the original unfiltered waveform data
- PSD shall be updated automatically when a valid window is selected

PSD calculation parameters:

- Method: Welch
- `window = hann`
- `detrend = linear`
- `nperseg = selected window full length`
- `noverlap = 50%`
- `scaling = density`

PSD display requirements:

- X-axis label: `Frequency (Hz)`
- Y-axis label: `PSD (dB rad^2/Hz)`
- Default visible frequency range: `1000 Hz ~ fs/2`
- X-axis shall support zooming and panning

### 4.6 Filtering

The software shall support optional filtering before time-domain plotting.

Filter enable rule:

- The user may choose filtered display or unfiltered display.

Supported filter types:

- Band-pass
- Low-pass

Default filter type:

- Band-pass

Filtering rules:

- Filtering applies only to the time-domain display
- PSD always uses the original unfiltered waveform data

Filter parameter input:

- For band-pass: low cutoff and high cutoff
- For low-pass: high cutoff only

### 4.7 Error Handling

The software shall provide user-visible status or warning messages for:

- Invalid directory
- No `.npz` files found
- Invalid or corrupted `.npz` file
- Missing required fields
- Failed `data_info` deserialization
- Invalid filter parameters
- Invalid selection window

## 5. Performance Requirements

The application targets high-rate waveform files. A typical file may contain approximately one minute of data at around `100 kHz` to `200 kHz` sample rate.

Performance targets:

- Initial open time for a single file: `< 1 s`
- Switching between files: `< 1 s`
- Zoom or drag refresh time: `< 1 s`

Implementation guidance:

- The application shall not rely on naive full-resolution redraw for all interactions.
- The main time-domain plot should use efficient rendering and downsampling strategies suitable for large arrays.

## 6. UI and Font Requirements

### 6.1 General UI

- Desktop application layout with a left control panel and a right analysis area
- Fixed two-plot layout on the right side

### 6.2 Font Requirements

- English text font: `Times New Roman`
- Chinese text font: `SimSun`

Plot requirements:

- Plot labels and tick text shall be in English
- Plot text shall use `Times New Roman`
- Tick font size shall be visually large enough for analysis use

## 7. Engineering Requirements

- Code shall be modular and maintainable.
- Repeated logic should be avoided.
- The application should remain lightweight and easy to extend.
- Source code shall be stored under `scripts`.
- Technical documentation shall be stored under `docs`.

## 8. Deliverables

The project deliverables shall include:

- A runnable desktop application implementation
- Source code under `scripts`
- Requirement and design documents under `docs`
- Basic usage documentation under `docs`

## 9. Acceptance Criteria

The software is accepted for v1.0 when:

- A directory can be selected and scanned for `.npz` files
- File pagination and sorting work as specified
- A file can be opened and plotted in the time domain
- Absolute time axis is displayed from filename parsing
- Y-axis auto/manual control works
- Zoom mode works
- Window PSD mode works
- PSD is computed and displayed from selected raw data
- Filtered and unfiltered time-domain display works
- `data_info` deserialization failure does not block normal file opening
