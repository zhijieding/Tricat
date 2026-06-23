"""
逐样本自适应 alpha — 基于 isotonic regression
==============================================
用 disagreement = ‖M - B‖₂ 作为特征，学习 α(x) 的单调映射。
比固定 3-bin 分桶更精细，也更不容易过拟合（isotonic 天然单调正则化）。

策略汇总：
  1. isotonic_alpha_oof   : 在 OOF 数据上用 per-sample pseudo-optimal α 作为 target
  2. apply_alpha_isotonic : 测试时根据 disagreement 查询 isotonic 预测 α(x)
  3. isotonic_alpha_cv    : 内部 K-fold 交叉验证版本（防止 isotonic 在 OOF 上微弱过拟合）
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression


def _row_norm(P):
    P = np.clip(P, 1e-12, None)
    return P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)


def _cdf3(P):
    P = _row_norm(P)
    return np.stack([P[:, 0], P[:, 0] + P[:, 1]], axis=1)


def _disagreement_l2(M, B):
    """逐样本 L2 距离 ‖M - B‖₂"""
    return np.sqrt(np.sum((M - B) ** 2, axis=1))


def _per_sample_optimal_alpha(B, M, y_true, clip_range=(0.0, 1.0)):
    """
    逐样本 pseudo-optimal alpha。
    对单个样本 i, RPS_i(α) = Σ_k (CB_k + α(CM_k - CB_k) - Co_k)^2
    是关于 α 的二次函数: a_i * α^2 + b_i * α + c_i
    最优: α_i* = -b_i / (2*a_i)   (若 a_i > 0)

    为了稳定性，对每个样本在小邻域内做 local optimal:
    """
    n = len(y_true)
    oh = np.zeros((n, 3))
    oh[np.arange(n), y_true] = 1.0
    CB, CM, Co = _cdf3(B), _cdf3(M), _cdf3(oh)
    D = CM - CB  # shape (n, 2)

    # 每样本: a_i = Σ_k D_ik^2,  b_i = 2 * Σ_k (CB_ik - Co_ik) * D_ik
    a = np.sum(D ** 2, axis=1)  # (n,)
    b = 2.0 * np.sum((CB - Co) * D, axis=1)  # (n,)

    alpha_star = np.where(a > 1e-12, -b / (2.0 * a), 0.0)
    alpha_star = np.clip(alpha_star, clip_range[0], clip_range[1])
    return alpha_star


def _per_sample_optimal_alpha_neighborhood(B, M, y_true, k=15, clip_range=(0.0, 1.0)):
    """
    稳定化版本：对每个样本，在 disagreement 相近的 k 个邻居上求局部最优 α。
    这样 target 更平滑，isotonic regression 学得更好。
    """
    n = len(y_true)
    oh = np.zeros((n, 3))
    oh[np.arange(n), y_true] = 1.0
    CB, CM, Co = _cdf3(B), _cdf3(M), _cdf3(oh)
    D = CM - CB

    dis = _disagreement_l2(M, B)
    sort_idx = np.argsort(dis)

    alpha_local = np.zeros(n, dtype=float)
    half_k = k // 2

    for rank, i in enumerate(sort_idx):
        lo = max(0, rank - half_k)
        hi = min(n, rank + half_k + 1)
        neighbors = sort_idx[lo:hi]

        D_nb = D[neighbors]
        CB_nb = CB[neighbors]
        Co_nb = Co[neighbors]

        a_nb = np.sum(D_nb ** 2)
        b_nb = 2.0 * np.sum((CB_nb - Co_nb) * D_nb)

        if a_nb > 1e-12:
            alpha_local[i] = np.clip(-b_nb / (2.0 * a_nb), clip_range[0], clip_range[1])
        else:
            alpha_local[i] = 0.0

    return alpha_local


def isotonic_alpha_oof(B_oof, M_oof, y_oof, method='neighborhood', k=25,
                       weight_by_disagreement=False):
    """
    在 OOF 数据上训练 isotonic regression: disagreement → α*(x)

    Parameters
    ----------
    B_oof : (n, 3) bookmaker probabilities
    M_oof : (n, 3) model probabilities
    y_oof : (n,) true labels
    method : 'pointwise' | 'neighborhood'
        pointwise: 直接用单样本 α_i*
        neighborhood: 用邻域平滑后的 α_i*（推荐，更稳定）
    k : int, neighborhood 半径
    weight_by_disagreement : bool
        True 时对 disagreement 大的样本给更大权重（它们对 RPS 影响更大）

    Returns
    -------
    iso_model : fitted IsotonicRegression
    alpha_global : float, global optimal α (fallback)
    stats : dict with diagnostics
    """
    n = len(y_oof)
    dis = _disagreement_l2(M_oof, B_oof)

    if method == 'neighborhood':
        alpha_targets = _per_sample_optimal_alpha_neighborhood(B_oof, M_oof, y_oof, k=k)
    else:
        alpha_targets = _per_sample_optimal_alpha(B_oof, M_oof, y_oof)

    # Global optimal alpha (for reference)
    oh = np.zeros((n, 3)); oh[np.arange(n), y_oof] = 1.0
    CB, CM, Co = _cdf3(B_oof), _cdf3(M_oof), _cdf3(oh)
    D = CM - CB
    a_global = np.sum(D ** 2)
    alpha_global = float(np.clip(np.sum((CB - CM) * (CB - Co)) / max(a_global, 1e-15), 0.0, 1.0))

    # Fit isotonic regression
    sample_weight = None
    if weight_by_disagreement:
        sample_weight = dis + 1e-6  # disagreement 越大权重越高

    iso_model = IsotonicRegression(
        y_min=0.0, y_max=1.0,
        out_of_bounds='clip',
        increasing=True  # 假设 disagreement 越大 → 应更信任模型 → α 更大
    )
    iso_model.fit(dis, alpha_targets, sample_weight=sample_weight)

    # Diagnostics
    alpha_pred_oof = iso_model.predict(dis)
    stats = {
        'alpha_global': alpha_global,
        'disagreement_quantiles': np.quantile(dis, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
        'alpha_pred_quantiles': np.quantile(alpha_pred_oof, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
        'alpha_target_quantiles': np.quantile(alpha_targets, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
        'n_samples': n,
    }

    return iso_model, alpha_global, stats


def isotonic_alpha_cv(B_oof, M_oof, y_oof, n_folds=5, method='neighborhood', k=25,
                      weight_by_disagreement=False):
    """
    K-fold 交叉验证版本：在 OOF 内部再做 CV 来防止 isotonic 过拟合。
    返回一个"平均"isotonic model（最终在全量 OOF 上重新 fit，但用 CV 选超参）。

    实际上 isotonic regression 的过拟合风险很小（单调约束），
    但如果样本量小（<1000），CV 版本更安全。
    """
    n = len(y_oof)
    dis = _disagreement_l2(M_oof, B_oof)

    if method == 'neighborhood':
        alpha_targets = _per_sample_optimal_alpha_neighborhood(B_oof, M_oof, y_oof, k=k)
    else:
        alpha_targets = _per_sample_optimal_alpha(B_oof, M_oof, y_oof)

    # CV to check if increasing=True is correct
    rng = np.random.default_rng(42)
    indices = np.arange(n)
    rng.shuffle(indices)
    fold_size = n // n_folds

    rps_increasing = 0.0
    rps_decreasing = 0.0

    for fold in range(n_folds):
        val_idx = indices[fold * fold_size: (fold + 1) * fold_size]
        tr_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])

        for direction, rps_acc in [('increasing', None), ('decreasing', None)]:
            iso_cv = IsotonicRegression(
                y_min=0.0, y_max=1.0,
                out_of_bounds='clip',
                increasing=(direction == 'increasing')
            )
            sw = (dis[tr_idx] + 1e-6) if weight_by_disagreement else None
            iso_cv.fit(dis[tr_idx], alpha_targets[tr_idx], sample_weight=sw)

            alpha_val = iso_cv.predict(dis[val_idx])
            # Compute RPS on val
            B_val = B_oof[val_idx]
            M_val = M_oof[val_idx]
            y_val = y_oof[val_idx]
            P_blend = (1.0 - alpha_val[:, None]) * B_val + alpha_val[:, None] * M_val
            P_blend = _row_norm(P_blend)

            oh_val = np.zeros((len(y_val), 3))
            oh_val[np.arange(len(y_val)), y_val] = 1.0
            cp = np.cumsum(P_blend, axis=1)[:, :2]
            co = np.cumsum(oh_val, axis=1)[:, :2]
            rps_val = float(np.mean(np.sum((cp - co) ** 2, axis=1)) / 2.0)

            if direction == 'increasing':
                rps_increasing += rps_val
            else:
                rps_decreasing += rps_val

    best_direction = 'increasing' if rps_increasing <= rps_decreasing else 'decreasing'

    # Final fit on all OOF data with best direction
    sample_weight = (dis + 1e-6) if weight_by_disagreement else None
    iso_final = IsotonicRegression(
        y_min=0.0, y_max=1.0,
        out_of_bounds='clip',
        increasing=(best_direction == 'increasing')
    )
    iso_final.fit(dis, alpha_targets, sample_weight=sample_weight)

    # Global alpha
    oh = np.zeros((n, 3)); oh[np.arange(n), y_oof] = 1.0
    CB, CM, Co = _cdf3(B_oof), _cdf3(M_oof), _cdf3(oh)
    D = CM - CB
    a_global = np.sum(D ** 2)
    alpha_global = float(np.clip(np.sum((CB - CM) * (CB - Co)) / max(a_global, 1e-15), 0.0, 1.0))

    stats = {
        'best_direction': best_direction,
        'rps_increasing_cv': rps_increasing / n_folds,
        'rps_decreasing_cv': rps_decreasing / n_folds,
        'alpha_global': alpha_global,
    }

    return iso_final, alpha_global, stats


def apply_alpha_isotonic(B, M, iso_model, alpha_global=None, floor_blend=0.02):
    """
    用训练好的 isotonic model 逐样本预测 α(x)，然后做加权融合。

    Parameters
    ----------
    B : (n, 3) bookmaker probs
    M : (n, 3) model probs
    iso_model : fitted IsotonicRegression
    alpha_global : float or None, fallback (not used if iso_model covers all range)
    floor_blend : float, 最小 model 权重，防止完全依赖 B365

    Returns
    -------
    P_fused : (n, 3) fused probabilities
    alpha_vec : (n,) per-sample alpha values used
    """
    dis = _disagreement_l2(M, B)
    alpha_vec = iso_model.predict(dis)

    # Optional: ensure minimum model contribution
    if floor_blend > 0:
        alpha_vec = np.maximum(alpha_vec, floor_blend)

    P_fused = (1.0 - alpha_vec[:, None]) * B + alpha_vec[:, None] * M
    P_fused = _row_norm(P_fused)

    return P_fused, alpha_vec


def isotonic_alpha_enhanced(B_oof, M_oof, y_oof, k=25,
                            shrink_to_global=0.1,
                            multi_feature=False):
    """
    增强版：
    1. neighborhood 平滑 target
    2. 可选 shrinkage：最终 α(x) = (1-λ)*iso(x) + λ*α_global
    3. 可选多特征（disagreement + entropy of B）— 此时改用 binning + isotonic

    Parameters
    ----------
    shrink_to_global : float in [0,1]
        0 = 纯 isotonic, 1 = 退化为 global α
    multi_feature : bool
        True 时同时考虑 B365 的 entropy 和 disagreement（分层 isotonic）

    Returns
    -------
    iso_model / (iso_low, iso_high) : fitted model(s)
    alpha_global : float
    stats : dict
    """
    n = len(y_oof)
    dis = _disagreement_l2(M_oof, B_oof)
    alpha_targets = _per_sample_optimal_alpha_neighborhood(B_oof, M_oof, y_oof, k=k)

    # Global alpha
    oh = np.zeros((n, 3)); oh[np.arange(n), y_oof] = 1.0
    CB, CM, Co = _cdf3(B_oof), _cdf3(M_oof), _cdf3(oh)
    D = CM - CB
    a_global = np.sum(D ** 2)
    alpha_global = float(np.clip(np.sum((CB - CM) * (CB - Co)) / max(a_global, 1e-15), 0.0, 1.0))

    # Shrink targets toward global
    alpha_targets_shrunk = (1.0 - shrink_to_global) * alpha_targets + shrink_to_global * alpha_global

    if not multi_feature:
        iso_model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip', increasing=True)
        iso_model.fit(dis, alpha_targets_shrunk, sample_weight=dis + 1e-6)
        stats = {
            'alpha_global': alpha_global,
            'shrink_to_global': shrink_to_global,
            'multi_feature': False,
        }
        return iso_model, alpha_global, stats
    else:
        # 分层: B365 entropy 高/低各自做一个 isotonic
        entropy_B = -np.sum(B_oof * np.log(np.clip(B_oof, 1e-12, 1.0)), axis=1)
        median_ent = np.median(entropy_B)
        mask_low = entropy_B <= median_ent
        mask_high = ~mask_low

        iso_low = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip', increasing=True)
        iso_high = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip', increasing=True)

        if mask_low.sum() > 20:
            iso_low.fit(dis[mask_low], alpha_targets_shrunk[mask_low],
                       sample_weight=dis[mask_low] + 1e-6)
        else:
            iso_low.fit(dis, alpha_targets_shrunk, sample_weight=dis + 1e-6)

        if mask_high.sum() > 20:
            iso_high.fit(dis[mask_high], alpha_targets_shrunk[mask_high],
                        sample_weight=dis[mask_high] + 1e-6)
        else:
            iso_high.fit(dis, alpha_targets_shrunk, sample_weight=dis + 1e-6)

        stats = {
            'alpha_global': alpha_global,
            'shrink_to_global': shrink_to_global,
            'multi_feature': True,
            'entropy_median': float(median_ent),
            'n_low': int(mask_low.sum()),
            'n_high': int(mask_high.sum()),
        }
        return (iso_low, iso_high, median_ent), alpha_global, stats


def apply_alpha_isotonic_enhanced(B, M, model_pack, alpha_global):
    """
    对 isotonic_alpha_enhanced 的输出做预测。

    model_pack: iso_model (single) 或 (iso_low, iso_high, median_ent) (multi_feature)
    """
    dis = _disagreement_l2(M, B)

    if isinstance(model_pack, tuple) and len(model_pack) == 3:
        iso_low, iso_high, median_ent = model_pack
        entropy_B = -np.sum(B * np.log(np.clip(B, 1e-12, 1.0)), axis=1)
        mask_low = entropy_B <= median_ent

        alpha_vec = np.zeros(len(B), dtype=float)
        if mask_low.any():
            alpha_vec[mask_low] = iso_low.predict(dis[mask_low])
        if (~mask_low).any():
            alpha_vec[~mask_low] = iso_high.predict(dis[~mask_low])
    else:
        alpha_vec = model_pack.predict(dis)

    alpha_vec = np.clip(alpha_vec, 0.0, 1.0)
    P_fused = (1.0 - alpha_vec[:, None]) * B + alpha_vec[:, None] * M
    P_fused = _row_norm(P_fused)
    return P_fused, alpha_vec
