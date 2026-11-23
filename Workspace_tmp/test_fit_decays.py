"""fit_decays (parallel) == Fit_decay row by row, on synthetic bi-exponential decays; timing per worker count.
python test_fit_decays.py [n_decays] [workers,...]"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flim_helpers import Fit_decay, fit_decays


def synth(n, nbin=128, Trep=12.5, seed=0):
    rng = np.random.default_rng(seed)
    dt = Trep / nbin
    t = np.arange(nbin) * dt
    irf = np.exp(-0.5 * ((t - 1.0) / 0.12) ** 2)
    out = []
    for _ in range(n):
        t1, t2, f1 = rng.uniform(0.4, 1.0), rng.uniform(2.0, 3.5), rng.uniform(0.3, 0.8)
        dec = f1 * np.exp(-t / t1) + (1 - f1) * np.exp(-t / t2)
        y = np.real(np.fft.ifft(np.fft.fft(irf) * np.fft.fft(dec)))
        out.append(rng.poisson(y / y.sum() * rng.uniform(2e3, 5e4)) + rng.poisson(2, nbin))
    return irf, np.array(out, float), Trep, dt


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    workers = [int(w) for w in sys.argv[2].split(',')] if len(sys.argv) > 2 else [8]
    irf, Y, Trep, dt = synth(n)
    t0 = time.perf_counter()
    ref = [Fit_decay(irf, y, Trep, dt, t_stop_ns=9.0)[:5] for y in Y[:4]]
    t_seq = (time.perf_counter() - t0) / 4
    print(f'sequential: {t_seq:.2f} s per fit')
    assert fit_decays(irf, Y[:0], Trep, dt) == []
    one = fit_decays(irf, Y[:4], Trep, dt, t_stop_ns=9.0, n_workers=1, verbose=False)
    for w in workers:
        t0 = time.perf_counter()
        par = fit_decays(irf, Y, Trep, dt, t_stop_ns=9.0, n_workers=w)
        el = time.perf_counter() - t0
        print(f'{w:2d} workers: {n} fits in {el:.1f} s  ({el / n:.3f} s per fit, x{t_seq * n / el:.1f} vs sequential)')
        for a, b, c in zip(ref, one, par[:4]):
            for u, v, x in zip(a, b, c):
                assert np.array_equal(np.asarray(u), np.asarray(v)) and np.array_equal(np.asarray(u), np.asarray(x))
    print('identical to sequential Fit_decay')
    print('ALL OK')
