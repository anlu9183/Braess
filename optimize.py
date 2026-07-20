import numpy as np
import time
TOL = 1e-12
'''
constructs eigenpair of path Laplacian on N vertices L_N
'''
def path_eig(N):
    a = np.arange(N)
    lam = 4.0 * np.sin(np.pi * a / (2 * N)) ** 2 # eigenvalues
    i = np.arange(N)[:, None]
    psi = np.cos(np.pi * a[None, :] * (i + 0.5) / N) # eigenvector
    return psi * np.sqrt(np.where(a == 0, 1.0 / N, 2.0 / N)), lam
'''
Compute the fraction of chords (non-edge vertex pairs) of an (m x n) grid graph
that exhibit Braess' paradox. A chord {u,v} is Braess-exhibiting iff the s-t
voltage drop satisfies V_uv < 0 with level gain k = csum[v] - csum[u] > 0.
Only the voltage vector phi = Lp[:,s] - Lp[:,t] is needed, so instead of forming
the full V x V pseudoinverse we contract the closed form to two columns:
    phi[i,j] = sum_{a,b} psi[i,a] xi[j,b] * G[a,b] * (psi[0,a]xi[0,b] - psi[m,a]xi[n,b])
Returns (fraction, braess_count, total_chords).
'''
def calc_BR(m, n, block=512):
    V = (m + 1) * (n + 1)
    rows, cols = np.divmod(np.arange(V), n + 1)
    csum = rows + cols
    psi, lam = path_eig(m + 1) # x axis eigenpairs
    xi,  mu  = path_eig(n + 1) # y axis eigenpairs
    eigenvalues = lam[:, None] + mu[None, :]; eigenvalues[0, 0] = 1.0
    G = 1.0 / eigenvalues; G[0, 0] = 0.0 # 1/eigenvalue, with constant mode -> 0
    # voltage at every vertex for unit current s=(0,0) -> t=(m,n), without building Lp
    w = G * (np.outer(psi[0], xi[0]) - np.outer(psi[m], xi[n]))  # [a,b]
    phi = (psi @ w @ xi.T).reshape(V)                            # phi[u] = Lp[u,s]-Lp[u,t]

    # total number of chords = all unordered vertex pairs minus existing grid edges
    edges = n * (m + 1) + m * (n + 1)          # horizontal + vertical grid edges
    total_chords = V * (V - 1) // 2 - edges

    braess = 0
    # process `block` rows at a time to utilize cache
    for u0 in range(0, V, block):
        u1 = min(u0 + block, V)
        Volt = phi[u0:u1, None] - phi[None, :]     # voltage drops V_uv (q>0 is sign-irrelevant)
        kblk = csum[None, :] - csum[u0:u1, None]   # level gains k
        braess += int(((Volt < -TOL) & (kblk > 0)).sum())

    fraction = braess / total_chords if total_chords else 0.0
    return fraction, braess, total_chords
'''
Search all grids (m x n) with 1 <= m <= n <= M.
Calls calc_BR for each grid, prints the max Braess-chord fraction over fixed m
as well as the global max fraction.
'''
def find_max(M=40, verbose=True):
    best = (0.0, None, None, None)  # global best (fraction, (m,n), count, total)
    t0 = time.time()
    for m in range(1, M + 1):
        best_for_m = (0.0, None, None, None)
        for n in range(m, M + 1):
            frac, count, total = calc_BR(m, n)
            if frac > best[0]:
                best = (frac, (m, n), count, total)
            if frac > best_for_m[0]:
                best_for_m = (frac, n, count, total)
        if verbose:
            elapsed = time.time() - t0
            frac, n_at = best_for_m[0], best_for_m[1]
            print(
                f"(elapsed {elapsed:5.0f}s, m={m}: "
                f"max Braess-chord fraction {frac:.7f} at n={n_at})",
                flush=True,
            )

    print("\n" + "=" * 60)
    frac, (m, n), count, total = best
    print(f"Max Braess-capable fraction = {frac:.10f}")
    print(f"  grid  {m} x {n}:  {count} of {total} chords exhibit Braess' paradox")
    return None
if __name__ == "__main__":
    find_max(M=100) # search all grids to M x M
