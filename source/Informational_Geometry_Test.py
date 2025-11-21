
import numpy as np
import matplotlib.pyplot as plt

def build_mutual_info(N, xi):
    idx = np.arange(N)
    d = np.abs(idx[:, None] - idx[None, :])
    return np.exp(-d / xi)

def build_kernel(I, alpha, jitter=1e-12, psd_project=False):
    K0 = np.power(I, alpha)
    K0 = 0.5 * (K0 + K0.T)
    if psd_project:
        w, v = np.linalg.eigh(K0)
        w = np.clip(w, 0, None)
        K_psd = (v * w) @ v.T
    else:
        K_psd = K0
    return K_psd + jitter * np.eye(K_psd.shape[0])

def normalize_kernel(K, eps=1e-12):
    d = np.sqrt(np.clip(np.diag(K), eps, None))
    return K / (d[:, None] * d[None, :])

def kernel_to_distances(K_norm, lP, eps=1e-12):
    Kc = np.clip(K_norm, eps, 1.0)
    R = -lP * np.log(Kc)
    np.fill_diagonal(R, 0.0)
    return 0.5 * (R + R.T)

def classical_mds_1d(D):
    N = D.shape[0]
    D2 = D**2
    J = np.ones((N, N)) / N
    H = np.eye(N) - J
    B = -0.5 * H @ D2 @ H
    w, v = np.linalg.eigh(B)
    idx = np.argmax(w)
    lam = max(w[idx], 0.0)
    x = v[:, idx] * np.sqrt(lam)
    if x[0] > x[-1]:
        x = -x
    return x

def procrustes_1d(x_src, x_tgt):
    std_src, std_tgt = np.std(x_src), np.std(x_tgt)
    if std_src < 1e-15 or std_tgt < 1e-15:
        return x_tgt.copy(), 1.0, 0.0, 1.0
    corr = np.corrcoef(x_src, x_tgt)[0, 1]
    scale = corr * (std_tgt / std_src)
    shift = np.mean(x_tgt) - scale * np.mean(x_src)
    x_aligned = scale * x_src + shift
    return x_aligned, scale, shift, corr

def stress1(d_embed, d_true):
    N = d_true.shape[0]
    iu = np.triu_indices(N, 1)
    num = np.sum((d_embed[iu] - d_true[iu])**2)
    den = np.sum(d_true[iu]**2)
    return np.sqrt(num / den)

def pairwise_dist(x):
    return np.abs(x[:, None] - x[None, :])

def min_triangle_violation(D):
    n = D.shape[0]
    vmin = np.inf
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i == j or j == k or i == k:
                    continue
                v = D[i, j] + D[j, k] - D[i, k]
                if v < vmin:
                    vmin = v
    return vmin

def main():
    N, alpha, xi = 10, 0.5, 2.02
    target_r0N = 1.797
    p_noise = 0.10
    I = build_mutual_info(N, xi)
    K0 = build_kernel(I, alpha)
    K = normalize_kernel(K0)
    step_coeff = alpha / xi
    lP = target_r0N / ((N - 1) * step_coeff)
    print(f"Calibrated lP = {lP:.6f} to match r0,{N-1} = {target_r0N}")
    R = kernel_to_distances(K, lP)
    x_embed = classical_mds_1d(R)
    x_true = np.linspace(-target_r0N / 2, target_r0N / 2, N)
    x_aligned, scale, shift, corr = procrustes_1d(x_embed, x_true)
    rel_rms = np.sqrt(np.mean((x_aligned - x_true)**2)) / np.std(x_true)
    D_embed = pairwise_dist(x_aligned)

    print(f"Relative RMS Error: {rel_rms:.4f}")
    print(f"Correlation: {corr:.4f}")
    
    s1 = stress1(D_embed, R)
    print(f"Stress-1: {s1:.4f}")
    
    tri_min = min_triangle_violation(R)
    print(f"Min triangle violation: {tri_min:.4e}")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(x_true, np.zeros_like(x_true), 'ko', label='True')
    plt.plot(x_aligned, np.zeros_like(x_aligned), 'rx', label='Emergent (Aligned)')
    for i in range(N):
        plt.plot([x_true[i], x_aligned[i]], [0, 0], 'g-', alpha=0.3)
    plt.yticks([])
    plt.title(f"Sim 1: Emergent 1D Geometry (N={N})")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
