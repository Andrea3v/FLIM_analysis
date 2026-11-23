"""vista_tiles.align_blocks on the 2026-11-06 session (real drops) and on synthetic sessions built from its Olympus
tiles (known drops / no drops).  Prints ALL OK when every check passes."""
import os, sys
import numpy as np
import tifffile
sys.path.insert(0, r'X:\FLIM_2026')
import vista_tiles as vt, olympus_oir as oo

D = r'D:\Peredox_FLIM\XYZTTile1_336_X0.4Y-2.8_Decay128_512x512_Z210_Ch1_T1_P1_S1'
BASE = 'XYZTTile1_336_X0.4Y-2.8_Decay128_512x512_Z210_Ch1_T1_P1_S1'
matl = oo.read_matl_info(oo.find_matl_dir(D))
ch = oo.pick_channel(oo.oir_metadata(matl['files'][0]))
oly = np.stack([oo.read_oir_channel(f, ch['id']) for f in matl['files']]).astype(np.float32).reshape(210, 512, 512)
iss = tifffile.imread(os.path.join(D, 'tiles_uint8', f'{BASE}_tiles_intensity.tif')).astype(np.float32).reshape(210, 512, 512)
ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f'  [{"ok" if cond else "FAIL"}] {name}')


# ── real session ──────────────────────────────────────────────────────────────
r = vt.align_blocks(iss, oly)
print(f'real: {len(r["empty"])} empty blocks, {len(r["missing"])} slots without data, discarded {r["discarded"].tolist()} '
      f'(NCC {np.round(r["ncc"][r["discarded"]], 2).tolist()}), shifted={r["shifted"]}')
miss = r['missing']
print('  missing per tile: ' + '  '.join(f't{t}: {[int(s % 35) for s in miss if s // 35 == t]}' for t in range(6)))
S = r['ncc_matrix']
kept = np.flatnonzero(r['slot_of_block'] >= 0)
at_gap = [int(b) for b in kept if b > 0 and r['slot_of_block'][b] - r['slot_of_block'][b - 1] > 1]
print(f'  kept blocks: NCC median {np.nanmedian(r["ncc"][kept]):.2f} min {np.nanmin(r["ncc"][kept]):.2f}; '
      f'blocks after a gap (kept): {[(b, round(float(r["ncc"][b]), 2)) for b in at_gap]}')
prev = np.load(r'C:\Users\Andrea\AppData\Local\Temp\claude\X--FLIM-2026\c8d93f44-2eb1-4357-84ee-b7563a8b7110\scratchpad\block_map.npy')
same = r['slot_of_block'][:185][kept[kept < 185]] == prev[kept[kept < 185]]
check(f'matches the earlier exploratory alignment ({same.mean():.0%} of kept blocks)', same.mean() > 0.97)
check('only block 31 (straddles the 11-plane gap) is discarded', r['discarded'].tolist() == [31])
check('25 empty blocks at the end of the file', np.array_equal(r['empty'], np.arange(185, 210)))
check('blocks 0-30 stay in place', np.array_equal(r['slot_of_block'][:31], np.arange(31)))

# ── synthetic: Olympus frames as ISS blocks (noisier), known drops ────────────
rng = np.random.default_rng(1)
noisy = rng.poisson(np.clip(oly, 0, None) * 0.3).astype(np.float32)          # ~3x fewer counts
drop = np.sort(rng.choice(np.arange(3, 207), 12, replace=False))
drop = np.unique(np.r_[drop, 60, 61, 62, 63, 64])                              # one 5-plane gap
keep = np.setdiff1d(np.arange(210), drop)
blocks = np.zeros_like(noisy)
blocks[:keep.size] = noisy[keep]
s = vt.align_blocks(blocks, oly)
print(f'synthetic drops: {drop.tolist()}')
print(f'  found missing: {s["missing"].tolist()}  discarded {s["discarded"].tolist()}')
exact = np.mean(s['slot_of_block'][:keep.size] == keep)
check(f'recovered slot of every recorded block ({exact:.1%} exact)', exact > 0.97)
check('missing slots = dropped planes (±1 plane)', all(np.min(np.abs(drop - m)) <= 1 for m in s['missing']))

# ── synthetic: no drops -> identity, nothing discarded ────────────────────────
n = vt.align_blocks(noisy, oly)
check('no drops: identity map', np.array_equal(n['block_of_slot'], np.arange(210)) and not n['shifted'])
check('no drops: nothing discarded', n['discarded'].size == 0)
print('ALL OK' if ok else 'SOME CHECKS FAILED')
