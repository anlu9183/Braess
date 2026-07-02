import numpy as np
import warnings
import matplotlib.pyplot as plt

from optimize import calc_BR

'''
Visual version of optimize.py.

Instead of printing the max Braess ratios (BR), this sweeps all grids
(m x n) with 1 <= m <= n <= M, stores every BR, and draws:
  1. a heatmap of BR over the (m, n) plane (lower triangle is unused),
  2. the best-over-n BR curve as a function of m,
with the global maximum marked on both.
'''


def compute_BR(M=40, verbose=True):
    '''Run calc_BR over all grids 1 <= m <= n <= M.

    Returns (BR, best) where BR[m-1, n-1] is the max ratio for that grid
    (NaN if Braess is impossible or n < m) and best is the global winner
    (ratio, (m, n), u, v, k).
    '''
    BR = np.full((M, M), np.nan)  # BR[m-1, n-1]
    best = (1.0, None, None, None, None)
    for m in range(1, M + 1):
        for n in range(m, M + 1):
            x = calc_BR(m, n)
            if x is None:
                continue
            ratio, u, v, k = x
            BR[m - 1, n - 1] = ratio
            if ratio > best[0]:
                best = (ratio, (m, n), u, v, k)
        if verbose:
            print(f"  m={m:3d} done", flush=True)
    return BR, best


def _highlight(ax, x, y, label=None):
    '''Mark a point with a clean solid dot.'''
    ax.scatter([x], [y], s=55, color="crimson", edgecolors="white",
               linewidths=1.0, zorder=5, label=label)


def plot_BR(BR, best, savepath="braess_ratios.png"):
    '''Draw the BR heatmap and the best-over-n curve, side by side.'''
    M = BR.shape[0]
    ratio, (gm, gn), u, v, k = best
    # BR is symmetric (an m x n grid equals an n x m grid); mirror the
    # computed upper triangle into the full plane for a contour plot.
    full = np.fmax(BR, BR.T)
    with warnings.catch_warnings():  # rows with no Braess are all-NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        best_per_m = np.nanmax(BR, axis=1)  # best over n >= m, as in find_max

    fig, (ax_map, ax_curve) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- filled contour of BR over the (m, n) plane ---
    axis = np.arange(1, M + 1)
    levels = np.linspace(1.0, np.nanmax(full), 21)
    cf = ax_map.contourf(axis, axis, full, levels=levels,
                         cmap="viridis", extend="min")
    ax_map.contour(axis, axis, full, levels=levels,
                   colors="white", linewidths=0.5, alpha=0.6)
    fig.colorbar(cf, ax=ax_map, label="max Braess ratio")
    _highlight(ax_map, gn, gm, label="global max")
    _highlight(ax_map, gm, gn)  # symmetric twin
    ax_map.set_xlabel("n")
    ax_map.set_ylabel("m")
    ax_map.set_title("Max Braess ratio over grid shape (linear contour)")
    ax_map.legend(loc="upper right")

    # --- best-over-n BR as a function of m ---
    ms = np.arange(1, M + 1)
    ax_curve.plot(ms, best_per_m, marker="o", ms=3, lw=1.2)
    _highlight(ax_curve, gm, ratio)
    ax_curve.annotate(
        f"global max {ratio:.6f}\n{gm}x{gn}, edge {u}->{v}, k={k}",
        xy=(gm, ratio), xytext=(34, 4), textcoords="offset points",
        ha="left", va="top", fontsize=9,
    )
    ax_curve.set_xlabel("m")
    ax_curve.set_ylabel("max Braess ratio over n >= m")
    ax_curve.set_title("Best Braess ratio for each m")
    ax_curve.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    print(f"\nGlobal Max Braess ratio = {ratio:.10f}")
    print(f"  grid  {gm} x {gn}  edge {u} -> {v}   level gain k = {k}")
    print(f"  figure saved to {savepath}")
    plt.show()


if __name__ == "__main__":
    BR, best = compute_BR(M=100)  # search all grids to M x M
    plot_BR(BR, best)
