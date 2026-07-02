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
Calculate the maximum BR over any combination of nondecreasing,
nonnegative affine latency functions on an (m x n) grid graph.

Builds the Laplacian pseudoinverse in closed form,
giving pairwise effective resistances and voltage drops.
Calculates worst-case BR for each pair in a single vectorized pass and finds max.

Returns None if Braess' Paradox is impossible, otherwise returns a tuple of
(max ratio, u coordinates, v coordinates, level gain k).
'''
def calc_BR(m, n, q=1.0, block=512):
    V = (m+1) * (n+1)
    rows, cols = np.divmod(np.arange(V), n + 1) 
    csum = rows + cols                          
    s, t = 0, V - 1                             

    psi, lam = path_eig(m + 1) # x axis eigenpairs
    xi,  mu  = path_eig(n + 1) # y axis eigenpairs
    eigenvalues = lam[:, None] + mu[None, :]; eigenvalues[0, 0] = 1.0
    G = 1.0 / eigenvalues; G[0, 0] = 0.0 # 1/eigenvalue, with constant mode -> 0

    # L_p[(i,j), (p,q)] = sum_{a,b} G[a,b] * psi[i,a] psi[p,a] * xi[j,b] xi[q,b]
    a_sum = np.einsum('ia,pa,ab->ipb', psi, psi, G, optimize=True) 
    Lp = np.einsum('jb,qb,ipb->ijpq', xi, xi, a_sum, optimize=True).reshape(V, V)

    dg = np.diag(Lp).copy()                      
    phi = Lp[:, s] - Lp[:, t]
    R_st = phi[s] - phi[t]                       

    best_ratio = -np.inf
    best_u = best_v = -1

    # process `block` rows at a time to utilize cache
    for u0 in range(0, V, block):
        u1 = min(u0 + block, V)
        Res  = dg[u0:u1, None] + dg[None, :] - 2.0 * Lp[u0:u1] # effective resistances
        Volt = q * (phi[u0:u1, None] - phi[None, :]) # voltage drops
        kblk = csum[None, :] - csum[u0:u1, None] # level gains
        possible = (Volt < -TOL) & (kblk > 0)
        if not possible.any():
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            denom  = q * kblk * R_st + q * (m + n) * Res - (m + n) * Volt
            ratios = np.where(possible, 1.0 - (Volt * kblk)/denom, -np.inf)
        best_idx = int(np.argmax(ratios))
        r = float(ratios.flat[best_idx])
        if r > best_ratio:
            best_ratio = r
            bi, vj = divmod(best_idx, ratios.shape[1])
            best_u, best_v = u0 + bi, vj

    if best_u < 0: # Braess Paradox Impossible
        return None
    k_uv = int(csum[best_v] - csum[best_u])
    u = (best_u // (n + 1), best_u % (n + 1))
    v = (best_v // (n + 1), best_v % (n + 1))
    return best_ratio, u, v, k_uv

'''
Search all grids (m x n) with 1 <= m <= n <= M 

Calls calc_BR for each grid, prints maximum BR over fixed m as well as global max BR.
'''
def find_max(M=40, verbose=True): 
    best = (1.0, None, None, None, None) # global best (ratio, (m,n), u, v, k)
    t0 = time.time() 
    for m in range(1, M + 1):
        best_for_m = (1.0, None, None, None, None) 
        for n in range(m, M+1):
            x = calc_BR(m,n) 
            if(x is None):
                continue
            ratio, u, v, k = x
            if ratio > best[0]: 
                best = (ratio, (m,n), u, v, k)
            if ratio > best_for_m[0]: 
                best_for_m = (ratio, n, u, v, k)

        if verbose:
            elapsed = time.time() - t0
            br, n_at = best_for_m[0], best_for_m[1]
            print(
                f"(elapsed {elapsed:5.0f}s, m={m}: "
                f"max BR {br:.7f} at n={n_at})",
                flush=True,
            )
            
    print("\n" + "=" * 60)
    ratio, (m, n), u, v, k = best
    print(f"Global Max Braess ratio = {ratio:.10f}")
    print(f"  grid  {m} x {n}  edge {u} -> {v}   level gain k = {k}")
    return None


if __name__ == "__main__":
    find_max(M = 100) # search all grids to M x M
