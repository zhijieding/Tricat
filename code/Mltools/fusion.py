import numpy as np
from sklearn.isotonic import IsotonicRegression
from .metrics import _row_norm, cdf3

def rps_optimal_alpha(B, M, y_true):
    n = len(y_true)
    oh = np.zeros((n,3)); oh[np.arange(n), y_true] = 1.0
    CB, CM, Co = cdf3(B), cdf3(M), cdf3(oh)
    D = CM - CB
    a = np.sum(D*D)
    if a <= 1e-15: return 0.0
    alpha = np.sum((CB - CM)*(CB - Co)) / a
    return float(np.clip(alpha, 0.0, 1.0))

def rps_optimal_alpha_binwise(B, M, y_true, pD_ref, bin_edges=None, shrink_lambda=10.0, min_bin=10):
    if bin_edges is None:
        bin_edges = np.array([0.0, 0.15, 0.25, 0.35, 0.45, 0.60, 1.0], dtype=float)
    else:
        bin_edges = np.asarray(bin_edges, dtype=float)
    bin_edges[0]  = 0.0
    bin_edges[-1] = 1.0
    nb = len(bin_edges) - 1

    alpha_global = rps_optimal_alpha(B, M, y_true)

    n = len(y_true)
    oh = np.zeros((n,3)); oh[np.arange(n), y_true] = 1.0
    CB, CM, Co = cdf3(B), cdf3(M), cdf3(oh)

    idx = np.digitize(np.clip(pD_ref, 0.0, 1.0), bin_edges, right=False) - 1
    idx = np.clip(idx, 0, nb-1)

    alpha_bins = np.zeros(nb, dtype=float)
    counts = np.zeros(nb, dtype=int)

    for b in range(nb):
        mask = (idx == b)
        counts[b] = int(mask.sum())
        if counts[b] < min_bin:
            alpha_bins[b] = alpha_global
            continue
        D = CM[mask] - CB[mask]
        a = np.sum(D*D)
        if a <= 1e-15:
            alpha_hat = alpha_global
        else:
            alpha_hat = np.sum((CB[mask] - CM[mask]) * (CB[mask] - Co[mask])) / a
            alpha_hat = float(np.clip(alpha_hat, 0.0, 1.0))
        n_bin = float(counts[b])
        alpha_bins[b] = (n_bin * alpha_hat + shrink_lambda * alpha_global) / (n_bin + shrink_lambda)

    print("[分桶 α*] 每桶样本数:", counts.tolist())
    return bin_edges, alpha_bins, alpha_global

def apply_alpha_binwise(B, M, pD_ref, bin_edges, alpha_bins):
    idx = np.digitize(np.clip(pD_ref, 0.0, 1.0), bin_edges) - 1
    idx = np.clip(idx, 0, len(alpha_bins)-1)
    alpha_vec = alpha_bins[idx]
    Pf = (1.0 - alpha_vec[:,None]) * B + alpha_vec[:,None] * M
    return _row_norm(Pf)


# ======================================================================
# 新方法: Isotonic α (Global + Residual Correction)
# 替代 3-bin 分桶，RPS 降低约 0.0004
# ======================================================================

def _disagreement_l2(M, B):
    return np.sqrt(np.sum((M - B)**2, axis=1))


def _neighbor_alpha_targets(B, M, y_true, feature, k=30):
    """邻域平滑的 per-sample pseudo-optimal α*"""
    n = len(y_true)
    oh = np.zeros((n, 3)); oh[np.arange(n), y_true] = 1.0
    CB, CM, Co = cdf3(B), cdf3(M), cdf3(oh)
    D = CM - CB
    sort_idx = np.argsort(feature)
    alpha_local = np.zeros(n)
    half_k = k // 2
    for rank, i in enumerate(sort_idx):
        lo = max(0, rank - half_k)
        hi = min(n, rank + half_k + 1)
        nb = sort_idx[lo:hi]
        D_nb = D[nb]; CB_nb = CB[nb]; Co_nb = Co[nb]
        a = np.sum(D_nb**2)
        b_val = 2.0 * np.sum((CB_nb - Co_nb) * D_nb)
        if a > 1e-12:
            alpha_local[i] = np.clip(-b_val / (2.0 * a), 0.0, 1.0)
    return alpha_local


def fit_isotonic_residual(B_oof, M_oof, y_oof, k=40, shrink_target=0.8):
    """
    训练 "Global α + Isotonic residual" 模型。

    步骤:
    1. 计算 global α*
    2. 做 global blend: P_global = (1-α)*B + α*M
    3. 以 P_global 为 base, 学习残差修正 δ(x)
       P_final = (1-δ)*P_global + δ*M
    4. δ 由 isotonic regression on disagreement=‖M-B‖₂ 给出
    5. 方向自动 CV 选择

    Parameters
    ----------
    B_oof, M_oof : (n, 3) OOF bookmaker/model probs
    y_oof : (n,) true labels
    k : neighborhood size for target smoothing
    shrink_target : shrinkage on residual targets (0.8 = conservative)

    Returns
    -------
    iso_residual : fitted IsotonicRegression
    alpha_global : float
    direction : bool (increasing or not)
    """
    n = len(y_oof)
    alpha_global = rps_optimal_alpha(B_oof, M_oof, y_oof)
    P_global = _row_norm((1 - alpha_global) * B_oof + alpha_global * M_oof)

    dis = _disagreement_l2(M_oof, B_oof)
    alpha_targets = _neighbor_alpha_targets(P_global, M_oof, y_oof, dis, k=k)
    alpha_targets = shrink_target * alpha_targets

    # 5-fold CV to pick direction
    rng_cv = np.random.default_rng(42)
    indices = np.arange(n)
    rng_cv.shuffle(indices)
    n_cv = 5
    fold_size = n // n_cv

    rps_inc, rps_dec = 0.0, 0.0
    for fold in range(n_cv):
        val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        tr_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
        for inc in [True, False]:
            iso = IsotonicRegression(y_min=0.0, y_max=0.5, out_of_bounds='clip', increasing=inc)
            iso.fit(dis[tr_idx], alpha_targets[tr_idx])
            delta_v = iso.predict(dis[val_idx])
            P_v = _row_norm((1 - delta_v[:, None]) * P_global[val_idx] + delta_v[:, None] * M_oof[val_idx])
            # vectorized RPS
            oh_v = np.zeros((len(val_idx), 3)); oh_v[np.arange(len(val_idx)), y_oof[val_idx]] = 1.0
            cp = np.cumsum(P_v, axis=1)[:, :2]
            co = np.cumsum(oh_v, axis=1)[:, :2]
            rps_v = float(np.mean(np.sum((cp - co) ** 2, axis=1)) / 2.0)
            if inc:
                rps_inc += rps_v
            else:
                rps_dec += rps_v

    best_dir = rps_inc <= rps_dec
    print(f"[Isotonic residual] CV: inc={rps_inc/n_cv:.8f} dec={rps_dec/n_cv:.8f} → "
          f"{'increasing' if best_dir else 'decreasing'}")
    print(f"[Isotonic residual] alpha_global = {alpha_global:.6f}")

    iso_residual = IsotonicRegression(y_min=0.0, y_max=0.5, out_of_bounds='clip', increasing=best_dir)
    iso_residual.fit(dis, alpha_targets)

    return iso_residual, alpha_global, best_dir


def apply_isotonic_residual(B, M, iso_residual, alpha_global):
    """
    应用 isotonic residual 融合。

    Parameters
    ----------
    B : (n, 3) bookmaker probs
    M : (n, 3) model probs
    iso_residual : fitted IsotonicRegression
    alpha_global : float

    Returns
    -------
    P_fused : (n, 3) fused probabilities
    alpha_effective : (n,) effective per-sample alpha
    """
    P_global = _row_norm((1 - alpha_global) * B + alpha_global * M)
    dis = _disagreement_l2(M, B)
    delta = iso_residual.predict(dis)
    P_fused = _row_norm((1 - delta[:, None]) * P_global + delta[:, None] * M)
    # effective alpha: (1-δ)*(1-α_g) weight on B, rest on M
    alpha_effective = alpha_global + delta * (1 - alpha_global)
    return P_fused, alpha_effective
