# FIPread Development Log

Purpose: keep a simple long-term update log for this project.

## 2026-03-26 14:00
- Added threshold filtering workflow for file list.
- Increased page size and improved paging UX.
- Added page jump and progress feedback.
- Problem solved: large-file browsing and quick file screening became easier.

## 2026-03-31 18:00
- Upgraded `scripts_svm` with visible-waveform audio playback/export.
- Added `Play / Stop / Replay`, `Audio Path`, and `Audio Downsample`.
- Problem solved: users can directly listen to and export visible waveform segments.

## 2026-04-14 21:29
- Merged `scripts` + `scripts_svm` into one program based on the `scripts_svm` branch.
- Added Plot 2 mode switch (`SVM Prediction` / `Short-Time Energy`).
- Merged `Display Controls` and `Short-Time Feature` into left-side tabs.
- Enabled manual width resize for left panel with splitter layout.
- Problem solved: one unified app now supports both feature views and better panel ergonomics.

## 2026-04-14 22:10
- Optimized UI layout and visual style for the left control panel and top header.
- Added top branding header with logo + centered title.
- Refactored control tabs to three sections: `Display Controls`, `Short-Time Feature`, and `Audio`.
- Moved audio controls into the dedicated `Audio` tab without changing audio behavior.
- Increased spacing and border contrast for better visual grouping across left-side modules.
- Standardized button visual feedback and set all button text to bold.
- Updated visible-time precision to 3 decimal places for label and input consistency.
- Problem solved: improved readability, clearer module separation, and more consistent interaction feedback.
