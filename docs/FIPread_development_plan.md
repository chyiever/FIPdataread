# FIPread Development Plan

## Phase 1. Requirements Consolidation

- Finalize requirement specification v1.0
- Confirm data protocol from sample `.npz`
- Freeze interaction rules, filter rules, PSD rules, and performance targets

## Phase 2. Architecture Setup

- Create `scripts/fipread` package structure
- Separate data access, signal processing, plotting helpers, and UI
- Define application state and file metadata models

## Phase 3. Core Feature Implementation

- Implement directory scan, sorting, and pagination
- Implement robust `.npz` loading with `data_info` degradation
- Implement filename-based absolute time parsing
- Implement time-domain plotting with pyqtgraph
- Implement manual and automatic Y-axis control
- Implement filter controls for band-pass and low-pass display

## Phase 4. Analysis Interaction

- Implement `Zoom` mode
- Implement `Window PSD` mode
- Implement short-window shading on the time-domain plot
- Implement PSD calculation from raw data with Welch
- Implement lower PSD panel update and view range initialization

## Phase 5. Verification and Documentation

- Run local smoke tests for import, file loading, filtering, and PSD calculation
- Verify example files can be scanned and opened
- Add user-facing run instructions
- Record known limitations and implementation notes

## Current Delivery Scope

The current delivery targets a usable v1.0 desktop application with:

- Stable file browsing
- High-volume waveform plotting using pyqtgraph downsampling features
- Deterministic interaction modes
- Graceful metadata degradation

## Risk Notes

- `.npz` compression can still limit absolute first-open performance compared with memory-mapped raw arrays
- Some machines may not have the required fonts installed
- If source files contain incompatible pickled metadata, only primary waveform analysis is guaranteed
