# FLIM Analysis

TCSPC-FLIM analysis pipeline for z-stack data: 3D intensity pre-processing, IRF correction, per-pixel mean-arrival-time lifetime maps, false-colour FLIM images, multi-exponential decay fitting, phasor analysis, and Cellpose 3D cell segmentation.

## Files

| File | Description |
|------|-------------|
| `AnalyzeTCSPC.ipynb` | Main analysis notebook — run cells in order |
| `flim_helpers.py` | Helper functions: `uplowthresh`, `Fit_decay`, `Calc_Phasor_Mat`, `phasor_ref_correction`, `read_fbs_metadata` |
| `FastFLIM_mod_AT0.py` | False-colour FLIM image generator (rainbow colormap, intensity modulation) |

## Analysis steps (notebook)

1. **Load data** — folder picker for z-stack TIFFs + single IRF TIFF; metadata read from `.fbs.xml`
2. **3D sumTCSPC pre-processing** — loads all planes, builds `vol_sum3d`; applies 3-D NLM denoising + per-slice rolling ball background subtraction → `vol_norm_3d` [0,1]; thresholds at `thresh_norm` → `mask3d`; saves `*_sumTCSPC_3Dfilt_norm.tif` and `*_mask3d.tif`
3. **Cellpose 3D segmentation** — segments cells in `vol_norm_3d` with `do_3D=True`; saves `*_cellpose_masks.tif` (uint16 labels)
4. **Mask preparation** — extracts middle-plane slice from `mask3d` → `maskTCSPC`, `maskFilt_bool`
5. **IRF preprocessing** — background subtraction, 1-D trace
6. **Peak realignment** — circularly shifts data and IRF so the fluorescence peak lands at 8% of the time window
7. **Mean arrival-time lifetime** (`tav = F − H`) — per-pixel lifetime map; bilateral filter for edge-preserving smoothing
8. **False-colour FLIM image** — `FastFLIM_mod_AT0` with auto or manual lifetime range; brightness modulated by NLM-filtered intensity
9. **Decay fitting** — `Fit_decay`: grid search + cascade Nelder-Mead, multi-exponential
10. **Phasor analysis** — `Calc_Phasor_Mat` with optional spatial binning and IRF reference correction
11. **Z-stack loop** — all planes processed using precomputed `mask3d` / `vol_norm_3d`; outputs multi-page TIFF stacks

## Key parameters (edit in `code-03`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Tcycle` | `12.5` ns | Laser repetition period (80 MHz) |
| `threshold` | `3` | Minimum photons per pixel for PMT mask |
| `thresh_norm` | `0.55` | Threshold on 3D-normalised sumTCSPC for `mask3d` [0–1] |
| `nlm_h` | `None` | NLM filter strength — `None` = auto (0.8 × estimated σ) |
| `nlm_patch_size` | `5` | NLM patch radius (voxels) |
| `nlm_patch_dist` | `6` | NLM search-window radius (voxels) |
| `nlm_fast` | `True` | NLM fast mode (~10× speedup) |
| `rb_radius` | `30` | Rolling ball radius (pixels, per z-slice) |
| `szBin` | `3` | Spatial binning for phasor (3×3 px blocks summed) |
| `szPhs` | `0.005` | Phasor histogram bin width (→ 201×201 density grid) |
| `limits` | `[nan, nan]` | Lifetime colormap range (ns); `nan` = auto from middle plane |

### Cellpose parameters (edit in segmentation cell)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cp_diameter` | `30` | Expected cell diameter in xy (pixels) — most important |
| `cp_anisotropy` | `3.0` | z step / xy pixel size (µm/µm) |
| `cp_model_type` | `'cyto3'` | Model: `'cyto3'`, `'cyto2'`, or `'nuclei'` |
| `cp_flow_threshold` | `0.4` | Raise to accept more cells |
| `cp_cellprob_thresh` | `0.0` | Lower = more cells detected |

## Environment

Conda env: `X:\conda\envs\FLIM`

| Package | Version |
|---------|---------|
| Python | 3.11 |
| numpy | 2.4.3 |
| matplotlib | 3.10.9 |
| scipy | 1.17.1 |
| scikit-image | 0.26.0 |
| tifffile | 2026.3.3 |
| imageio | 2.37.3 |
| cellpose | 4.1.1 |
| PyWavelets | 1.8.0 |
| tkinter | (stdlib) |
