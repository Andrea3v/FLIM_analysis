"""
FastFLIM_mod_AT0.py
Python translation of FastFLIM_mod_AT0.m

Creates a false-colour FLIM image by mapping per-pixel lifetime values onto a
wavelength-inspired rainbow colormap and modulating brightness by photon count.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def FastFLIM_mod_AT0(filename, tag, tim, thresh, lims, show=True):
    """
    Create a false-colour FLIM image from photon-count and lifetime maps.

    Parameters
    ----------
    filename : str or None
        Base path for saving (no extension). Pass None to skip saving.
    tag      : ndarray (nx, ny [, nch])
        Photon count per pixel.
    tim      : ndarray (nx, ny [, nch])
        Mean arrival-time / lifetime per pixel (ns).
    thresh   : float
        Minimum photon count to display a pixel.
    lims     : array-like [lo, hi]
        Lifetime colormap range (ns).

    Returns
    -------
    im : ndarray (nx, ny, 3)
        False-colour RGB image (last channel if input is multichannel).
    A  : ndarray (n_colors, 10, 3)
        Colorbar strip.
    """
    lims = np.asarray(lims, dtype=float)

    # ── 1. Threshold ──────────────────────────────────────────────────────────
    tim = tim.copy().astype(float)
    tim[tag < thresh] = 0.0
    tim = np.maximum(tim, 0.0)

    # ── 2. Intensity: power-law scaled photon count ───────────────────────────
    intens = tag.astype(float) ** 0.8
    intens = intens / intens.max()

    # ── 3. Rainbow colormap (wavelength-inspired) ─────────────────────────────
    # Keypoints: wavelength (nm) → RGB
    #   399 nm black  — used for masked / below-range pixels
    #   440 nm blue
    #   485 nm cyan
    #   540 nm green
    #   580 nm yellow
    #   610 nm orange
    #   630 nm red
    #   699 nm white
    xl = np.array([399, 440, 485, 540, 580, 610, 630, 699], dtype=float)
    yl = np.array([
        [0,    0,    0   ],
        [0,    0,    1   ],
        [0,    1,    1   ],
        [0,    1,    0   ],
        [1,    1,    0   ],
        [1,    0.65, 0   ],
        [1,    0,    0   ],
        [1,    1,    1   ],
    ])
    lambdas  = np.arange(399, 700, 3, dtype=float)          # 101 values
    spectrum = np.column_stack([
        np.interp(lambdas, xl, yl[:, i]) for i in range(3)
    ])                                                        # (101, 3)
    n_colors = spectrum.shape[0]                              # 101

    # ── 4. Uniform channel loop ────────────────────────────────────────────────
    if tim.ndim == 2:
        tim    = tim[:, :, np.newaxis]
        intens = intens[:, :, np.newaxis]

    im = None
    for ch in range(tim.shape[2]):

        tmp = tim[:, :, ch]
        val = tmp.ravel().astype(float)

        val[np.isnan(val)]  = 0.0
        val[val < lims[0]]  = 0.0
        val[val > lims[1]]  = lims[1]

        # Map lifetime → colormap index (0-based)
        #
        # MATLAB used: k = 2 + round((val-lo)/(hi-lo) * (N-2));  k(k<2) = 1
        # Index 1 (MATLAB) = index 0 (Python) = black → masked / out-of-range pixels
        # Index 2 (MATLAB) = index 1 (Python) → val == lo  (start of colour range)
        # Index N (MATLAB) = index N-1 (Python) → val == hi (white)
        #
        # Pixels set to val=0 produce norm < 0 → k < 1 → clipped to 0 (black).
        norm = (val - lims[0]) / (lims[1] - lims[0])
        k    = np.round(1.0 + norm * (n_colors - 2)).astype(int)
        k    = np.clip(k, 0, n_colors - 1)

        rgb = spectrum[k].reshape(tag.shape[0], tag.shape[1], 3)
        rgb = rgb * intens[:, :, ch, np.newaxis]

        im = rgb  # retain last channel (matches MATLAB behaviour)

        # ── display ───────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='k')
        ax.imshow(im, aspect='equal', origin='upper')
        ax.set_facecolor('black')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        cmap_listed = mcolors.ListedColormap(spectrum)
        norm_obj    = mcolors.Normalize(vmin=lims[0], vmax=lims[1])
        sm = plt.cm.ScalarMappable(cmap=cmap_listed, norm=norm_obj)
        sm.set_array([])

        tick_vals   = np.arange(lims[0], lims[1] + 1e-9, 0.5)
        tick_labels = [f'{v:.1f}' for v in tick_vals]

        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels(tick_labels)
        cbar.set_label('lifetime / ns', color='white', fontsize=9)
        cbar.ax.tick_params(direction='out', labelsize=9,
                            colors='white', length=4)
        cbar.outline.set_visible(False)

        plt.tight_layout()

        if filename:
            suffix = f'_ch{ch}' if tim.shape[2] > 1 else ''
            plt.savefig(filename + suffix + '_LTimg.png',
                        dpi=150, bbox_inches='tight', facecolor='k')

        if show:
            plt.show()
        else:
            plt.close(fig)

    # ── 5. Colorbar strip (n_colors × 10 × 3) ─────────────────────────────────
    A = np.repeat(spectrum[:, np.newaxis, :], 10, axis=1)

    return im, A
