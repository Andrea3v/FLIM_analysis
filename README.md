# FLIM Analysis

TCSPC-FLIM analysis pipeline: IRF correction, per-pixel mean-arrival-time lifetime maps, false-colour FLIM images, multi-exponential decay fitting, and phasor analysis.

## Files

| File | Description |
|------|-------------|
| `AnalyzeTCSPC.ipynb` | Main analysis notebook — run cells in order |
| `flim_helpers.py` | Helper functions: `uplowthresh`, `Fit_decay`, `Calc_Phasor_Mat`, `phasor_ref_correction` |
| `FastFLIM_mod_AT0.py` | False-colour FLIM image generator (rainbow colormap, intensity modulation) |

## Analysis steps (notebook)

1. **Load data** — TIFF picker for IRF and TCSPC image, percentile-based display thresholds
2. **Mask preparation** — photon-count threshold mask (`maskPixel`, `maskTCSPC`)
3. **IRF preprocessing** — background subtraction
4. **Peak realignment** — circularly shift data and IRF so the fluorescence peak lands at 8% of the time window
5. **Mean arrival-time lifetime** (`tav = F − H`) — per-pixel lifetime map
6. **False-colour FLIM image** — `FastFLIM_mod_AT0` with auto or manual lifetime range
7. **Decay fitting** — `Fit_decay`: grid search + cascade Nelder-Mead, multi-exponential
8. **Phasor analysis** — `Calc_Phasor_Mat` with optional spatial binning (`szBin`) and reference correction (`phasor_ref_correction`)

## Key parameters (edit in `code-03`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Tcycle` | `12.5` ns | Laser repetition period (80 MHz) |
| `threshold` | `3` | Minimum photons per pixel (inclusive) |
| `szBin` | `3` | Spatial binning for phasor (3×3 px blocks summed) |
| `szPhs` | `0.005` | Phasor histogram bin width (→ 201×201 density grid) |
| `limits` | `[nan, nan]` | Lifetime colormap range (ns); `nan` = auto |

## Environment

| Package | Version |
|---------|---------|
| Python | 3.11.2 |
| numpy | 1.23.2 |
| matplotlib | 3.7.0 |
| scipy | 1.10.1 |
| tifffile | 2024.12.12 |
| tkinter | 8.6 |
