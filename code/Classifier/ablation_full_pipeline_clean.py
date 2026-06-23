# -*- coding: utf-8 -*-
"""
Ablation Full Pipeline (CLEAN) — exp0..exp3 + Plots
===================================================
- 保持双头结构 + KD + 校准 + t,b + BPCF + 决策层。
- 产物：
  * comparison_summary.csv
  * comparison_summary_by_split.csv（df2223/df2324/ALL）
  * probs_exp*.csv, prf_exp*.csv, confmat_exp*.csv
  * 图：one_figure_comparison.png, rps_acc.png, f1_grouped.png,
       pred_distribution.png, rps_by_split.png, acc_by_split.png,
       f1_radar_df2223.png, f1_radar_df2324.png, f1_radar_ALL.png

运行示例：
python ablation_full_pipeline_clean.py \
  --data_dir "/mnt/train-decision-agent/dingzhijie/Trident-Cat code modification/数据/赛季数据" \
  --work_dir "/mnt/train-decision-agent/dingzhijie/Trident-Cat code modification/数据" \
  --out_dir  "/mnt/train-decision-agent/dingzhijie/Trident-Cat code modification/cat/ablation_4step"
"""

import os
import argparse
import random
import numpy as np
import pandas as pd

# ---------- Determinism ----------
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
rng = np.random.default_rng(42)

# ---------- Hyperparams (match original structure) ----------
BEST_SHARED = dict(
    iterations=816,
    depth=7,
    learning_rate=0.020131753910754745,
    l2_leaf_reg=14.294786288916372,
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
    loss_function='Logloss',
    bootstrap_type='Bayesian',
    bagging_temperature=0.44665711182112733,
    random_strength=0.7330774185074562,
    rsm=1.0,
    thread_count=1,
    task_type='CPU',
)

LAM_A = 0.559107646607841
LAM_B = 0.6659173145855906

TB = dict(A_t=1.050, A_b=-0.060, B_t=1.120, B_b=0.400)

# BPCF 参数 (Bounded Per-class Calibrated Fusion)
ALPHA_H = 0.42
ALPHA_D = 0.45
ALPHA_A = 0.93
ENSEMBLE_W_PLATT = 0.67
DEV_CAP = 0.065

DECISION = dict(alphaH=1.10, alphaD=1.30, alphaA=1.10, tauH=0.60, tauD=0.20, tauA=0.30, gamma=0.00)

# ---------- Columns ----------
FEATURE_COLS = [
    'Hform','Aform','Hst','ASt','HSTKPP','ASTKPP',
    'HGKPP','AGKPP','HCKPP','ACKPP','HAttack','AAttack',
    'HMidField','AMidField','HDefence','ADefense','HOverall','AOverall',
    'HTDG','ATDG','HStWeighted','AStWeighted','FormDifferential',
    'StDifferential','STKPP','GKPP','CKPP','RelAttack','RelMidField',
    'RelDefense','RelOverall','GDDifferential','StWeightedDifferential',
    'HS','AS','HF','AF','FDifferential','SDifferential'
]
KEEP_COLS = FEATURE_COLS + ['Date','HomeTeam','AwayTeam','FTR','B365H','B365D','B365A']

# ---------- Math utils ----------
def _row_norm(P):
    P = np.clip(P, 1e-12, None)
    s = P.sum(axis=1, keepdims=True)
    s = np.clip(s, 1e-12, None)
    return P / s

def _logit_clip(p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return np.log(p/(1-p))

def _sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def odds_to_probs(arr3):
    arr = np.clip(arr3.astype(float), 1e-12, None)
    p = 1.0 / arr
    return _row_norm(p)

def calc_rps(y_true, y_prob, n_class=3):
    y_true = np.asarray(y_true, dtype=int)
    P = np.asarray(y_prob, dtype=float)
    P = np.clip(P, 1e-12, None)
    P = _row_norm(P)
    n = len(y_true)
    if n == 0:
        return float('nan')
    rps_sum = 0.0
    for i in range(n):
        pr = P[i]
        oh = np.zeros(n_class); oh[int(y_true[i])] = 1.0
        cp = np.cumsum(pr); co = np.cumsum(oh)
        rps_sum += np.sum((cp - co) ** 2) / (n_class - 1)
    return rps_sum / max(n, 1)

# ---------- IO & cleaning ----------
def parse_date_safe(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, format='%d/%m/%Y', errors='coerce')
    for fmt in ['%d/%m/%y','%Y-%m-%d','%Y/%m/%d','%m/%d/%Y','%m-%d-%Y']:
        m = d.isna()
        if m.any():
            d2 = pd.to_datetime(s[m], format=fmt, errors='coerce')
            d.loc[m] = d2
    m = d.isna()
    if m.any():
        d3 = pd.to_datetime(s[m], errors='coerce', dayfirst=False)
        d.loc[m] = d3
    return d

def read_and_prepare(data_dir: str, work_dir: str):
    na = ['', ' ', 'NA', 'N/A', 'na', 'NaN']
    train_files = ['1415.csv','1516.csv','1617.csv','1718.csv','1819.csv','1920.csv','2021.csv','2122.csv']
    test_files  = ['2223.csv','2324.csv']

    train_out = os.path.join(work_dir, 'train'); os.makedirs(train_out, exist_ok=True)
    test_out  = os.path.join(work_dir, 'test');  os.makedirs(test_out, exist_ok=True)

    def _clean_and_save(src_folder, files, dst_folder):
        for file in files:
            fp = os.path.join(src_folder, file)
            if not os.path.exists(fp):
                print(f"[WARN] Missing file: {fp} (skipped)")
                continue
            df = pd.read_csv(fp, encoding='utf-8', na_values=na, keep_default_na=True)
            if 'FTR' in df.columns:
                df['FTR'] = df['FTR'].map({'H':0,'D':1,'A':2})
            df = df[[c for c in KEEP_COLS if c in df.columns]].copy()
            df = df.dropna(subset=[c for c in KEEP_COLS if c in df.columns], how='any')
            if 'Date' in df.columns:
                df['Date'] = parse_date_safe(df['Date'])
                df = df.dropna(subset=['Date']).sort_values('Date', kind='mergesort')
            df['src'] = f"df{file.split('.')[0]}"
            out = os.path.join(dst_folder, f"df{file.split('.')[0]}.csv")
            df.to_csv(out, index=False, encoding='utf-8')

    _clean_and_save(data_dir, train_files, train_out)
    _clean_and_save(data_dir, test_files,  test_out)

    def _cat(folder):
        parts = []
        for f in sorted(os.listdir(folder)):
            if f.endswith('.csv'):
                parts.append(pd.read_csv(os.path.join(folder, f), encoding='utf-8'))
        if parts:
            return pd.concat(parts, ignore_index=True).fillna(0)
        return pd.DataFrame()

    train = _cat(train_out)
    test  = _cat(test_out)

    if train.empty or test.empty:
        raise RuntimeError("Training or test data is empty — please check --data_dir and CSV presence.")

    if 'src' not in train.columns: train['src'] = 'train_all'
    if 'src' not in test.columns:  test['src']  = 'ALL'

    return train, test

# ---------- Modeling (CatBoost required) ----------
try:
    from catboost import CatBoostClassifier
except Exception as e:
    raise ImportError("CatBoost is required for this script to preserve your original results. "
                      "Please install: pip install catboost") from e

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def _build_soft_dataset(X, y_hard, soft_pos, lam=0.7):
    y_smooth = (1 - lam) * y_hard + lam * soft_pos
    X_dup = np.repeat(X, 2, axis=0)
    y_dup = np.empty(2 * len(y_hard), dtype=int)
    w_dup = np.empty(2 * len(y_hard), dtype=float)
    y_dup[0::2] = 1; w_dup[0::2] = np.clip(y_smooth, 1e-6, 1-1e-6)
    y_dup[1::2] = 0; w_dup[1::2] = np.clip(1 - y_smooth, 1e-6, 1-1e-6)
    return X_dup, y_dup, w_dup

def train_AB_heads(X_train_full, y_train_full, train_df, use_kd: bool):
    split = int(0.9 * len(X_train_full))
    X_fit, y_fit = X_train_full[:split], y_train_full[:split]
    X_cal, y_cal = X_train_full[split:], y_train_full[split:]

    train_b365 = odds_to_probs(train_df[['B365H','B365D','B365A']].values)
    b365_fit, b365_cal = train_b365[:split], train_b365[split:]

    # Head A (Draw vs Non-draw)
    yA_fit = (y_fit == 1).astype(int)
    if use_kd:
        softA_fit = b365_fit[:, 1]
        XA_dup, yA_dup, wA_dup = _build_soft_dataset(X_fit, yA_fit, softA_fit, lam=LAM_A)
    else:
        XA_dup, yA_dup, wA_dup = _build_soft_dataset(X_fit, yA_fit, yA_fit, lam=0.0)
    model_A = CatBoostClassifier(**BEST_SHARED)
    model_A.fit(XA_dup, yA_dup, sample_weight=wA_dup, verbose=False)

    # Head B (Home vs Away | Non-draw)
    maskB_fit = (y_fit != 1)
    XB_fit_raw = X_fit[maskB_fit]
    yB_fit_hard = (y_fit[maskB_fit] == 0).astype(int)
    if use_kd:
        teacher_fit_ha = b365_fit[maskB_fit][:, [0,2]]
        teacher_fit_ha = teacher_fit_ha / np.clip(teacher_fit_ha.sum(axis=1, keepdims=True), 1e-12, None)
        softB_pos_fit  = teacher_fit_ha[:, 0]
        XB_dup, yB_dup, wB_dup = _build_soft_dataset(XB_fit_raw, yB_fit_hard, softB_pos_fit, lam=LAM_B)
    else:
        XB_dup, yB_dup, wB_dup = _build_soft_dataset(XB_fit_raw, yB_fit_hard, yB_fit_hard, lam=0.0)
    model_B = CatBoostClassifier(**BEST_SHARED)
    model_B.fit(XB_dup, yB_dup, sample_weight=wB_dup, verbose=False)

    return (model_A, model_B), (X_fit, y_fit, X_cal, y_cal), (b365_fit, b365_cal)

def calibrate_heads(models, X_cal, y_cal, do_calibration: bool):
    model_A, model_B = models
    pD_cal_raw = model_A.predict_proba(X_cal)[:, 1]
    yA_cal = (y_cal == 1).astype(int)

    def calibrate_A_fn(p): return np.clip(p, 1e-12, 1-1e-12)
    if do_calibration and len(np.unique(yA_cal)) >= 2:
        isoA = IsotonicRegression(out_of_bounds='clip').fit(pD_cal_raw, yA_cal)
        def calibrate_A_fn(p): return np.clip(isoA.predict(p), 1e-12, 1-1e-12)

    maskB_cal = (y_cal != 1)
    pHnD_cal_raw = model_B.predict_proba(X_cal[maskB_cal])[:, 1]
    yB_cal = (y_cal[maskB_cal] == 0).astype(int)

    class _IdentityLR:
        def predict_proba(self, z):
            z = z.ravel(); p = 1.0/(1.0+np.exp(-z))
            return np.vstack([1-p, p]).T

    def calibrate_B_fn(p): return np.clip(p, 1e-12, 1-1e-12)
    if do_calibration:
        if len(np.unique(yB_cal)) <= 1:
            _ = _IdentityLR()  # no-op, keep identity
        else:
            z = _logit_clip(pHnD_cal_raw).reshape(-1,1)
            lrB = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000, class_weight='balanced').fit(z, yB_cal)
            def calibrate_B_fn(p):
                zz = _logit_clip(p).reshape(-1,1)
                return np.clip(lrB.predict_proba(zz)[:,1], 1e-12, 1-1e-12)

    return calibrate_A_fn, calibrate_B_fn

def assemble_prob_3class(models, calibrators, X, apply_tb: bool):
    model_A, model_B = models
    calib_A, calib_B = calibrators
    pD_raw   = model_A.predict_proba(X)[:, 1]
    pHnD_raw = model_B.predict_proba(X)[:, 1]
    pD   = calib_A(pD_raw)
    pHnD = calib_B(pHnD_raw)
    if apply_tb:
        pD   = _sigmoid(_logit_clip(pD)   * TB['A_t'] + TB['A_b'])
        pHnD = _sigmoid(_logit_clip(pHnD) * TB['B_t'] + TB['B_b'])
    pH = (1 - pD) * pHnD
    pA = (1 - pD) * (1 - pHnD)
    return _row_norm(np.column_stack([pH, pD, pA]))

def apply_bpcf(B, M, train_b365, y_train, n_skip):
    """BPCF: Per-class Logit Blend + Platt Ensemble + Deviation Cap"""
    # Platt scaling (fit on last 6 seasons of training data)
    platt_models = []
    for c in range(3):
        z = _logit_clip(train_b365[n_skip:, c]).reshape(-1, 1)
        yc = (y_train[n_skip:] == c).astype(int)
        lr_c = LogisticRegression(C=1e4, solver='lbfgs', max_iter=1000).fit(z, yc)
        platt_models.append(lr_c)

    # B365 Platt
    P_platt = np.zeros_like(B)
    for c in range(3):
        P_platt[:, c] = platt_models[c].predict_proba(_logit_clip(B[:, c]).reshape(-1, 1))[:, 1]
    P_platt = _row_norm(P_platt)

    # Per-class logit blend
    alpha_vec = np.array([ALPHA_H, ALPHA_D, ALPHA_A])
    logit_bl = (1 - alpha_vec) * _logit_clip(B) + alpha_vec * _logit_clip(M)
    P_logit_blend = _row_norm(_sigmoid(logit_bl))

    # Ensemble
    P_ensemble = _row_norm(ENSEMBLE_W_PLATT * P_platt + (1 - ENSEMBLE_W_PLATT) * P_logit_blend)

    # Deviation cap
    diff_p = P_ensemble - B
    dist = np.sqrt(np.sum(diff_p**2, axis=1, keepdims=True))
    scale = np.where(dist > DEV_CAP, DEV_CAP / np.clip(dist, 1e-12, None), 1.0)
    Pf = _row_norm(np.clip(B + diff_p * scale, 1e-12, None))
    return Pf

def classwise_temp(P, aH=1.0, aD=1.0, aA=1.0):
    Z = np.log(np.clip(P,1e-12,1-1e-12))
    Z = np.column_stack([Z[:,0]*aH, Z[:,1]*aD, Z[:,2]*aA])
    Q = np.exp(Z); return _row_norm(Q)

def predict_with_params(P_dec, tau_H, tau_D, tau_A, gamma, m_draw=0.05):
    S = np.column_stack([P_dec[:,0]/tau_H, P_dec[:,1]/tau_D, P_dec[:,2]/tau_A])
    S_max = np.maximum(S[:, 0], S[:, 2])
    is_draw = (S[:, 1] + m_draw >= S_max) & (P_dec[:, 1] >= tau_D)
    pred = np.full(len(P_dec), -1, dtype=int)
    pred[is_draw] = 1
    idx = ~is_draw
    Ph, Pa = P_dec[idx,0], P_dec[idx,2]
    denom = np.clip(Ph + Pa, 1e-12, None)
    p_home_cond = Ph / denom
    pred_nd = np.where(p_home_cond >= 0.5 + gamma, 0,
                np.where(p_home_cond <= 0.5 - gamma, 2,
                         np.where(S[idx,0] > S[idx,2], 0, 2)))
    pred[idx] = pred_nd
    return pred

# ---------- Plotting (matplotlib; one figure per chart; no explicit colors) ----------
import matplotlib.pyplot as plt

def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def one_figure_plot(out_dir):
    sum_csv = os.path.join(out_dir, 'comparison_summary.csv')
    if not os.path.exists(sum_csv):
        return
    df_sum = pd.read_csv(sum_csv)
    order = ['exp0','exp1','exp2','exp3']
    df_sum['Step'] = pd.Categorical(df_sum['Step'], categories=order, ordered=True)
    df_sum = df_sum.sort_values('Step')
    steps = df_sum['Step'].tolist()
    rps = [ _safe_float(v) for v in df_sum['RPS'].tolist() ]
    acc = [ _safe_float(v) for v in df_sum['ACC'].tolist() ]
    # F1 & predicted share
    f1_H, f1_D, f1_A = [], [], []
    shares_H, shares_D, shares_A = [], [], []
    for st in steps:
        prf_fp = os.path.join(out_dir, f'prf_{st}.csv')
        if os.path.exists(prf_fp):
            prf = pd.read_csv(prf_fp, index_col=0)
            f1_H.append(_safe_float(prf.loc['Home wins','F1-score']))
            f1_D.append(_safe_float(prf.loc['Draws','F1-score']))
            f1_A.append(_safe_float(prf.loc['Away wins','F1-score']))
        else:
            f1_H.append(np.nan); f1_D.append(np.nan); f1_A.append(np.nan)
        cm_fp = os.path.join(out_dir, f'confmat_{st}.csv')
        if os.path.exists(cm_fp):
            cm_df = pd.read_csv(cm_fp, index_col=0)
            col_sums = cm_df.sum(axis=0).values.astype(float)
            tot = col_sums.sum() if col_sums.sum()>0 else 1.0
            share = col_sums / tot
            shares_H.append(share[0]); shares_D.append(share[1]); shares_A.append(share[2])
        else:
            shares_H.append(np.nan); shares_D.append(np.nan); shares_A.append(np.nan)

    x = np.arange(len(steps))
    fig = plt.figure(figsize=(12, 7), dpi=150)
    ax = fig.add_subplot(111)
    bars = ax.bar(x, rps, width=0.6, alpha=0.9, linewidth=1.2)
    for i, v in enumerate(rps):
        if np.isfinite(v):
            ax.text(x[i], v, f'{v:.6f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('RPS (lower is better)')
    ax.set_xticks(x, steps)
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(x, acc, marker='o', linewidth=2.2, label='ACC')
    ax2.plot(x, f1_H, marker='^', linewidth=1.8, label='F1 Home')
    ax2.plot(x, f1_D, marker='s', linewidth=1.8, label='F1 Draw')
    ax2.plot(x, f1_A, marker='v', linewidth=1.8, label='F1 Away')
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel('ACC / F1')
    ax.set_title('Ablation exp0..exp3 — RPS vs ACC + F1(H/D/A) — Predicted Distribution Table', fontsize=13)
    ax2.legend(loc='upper right', framealpha=0.9)

    # table
    lines = ["Step | Pred H | Pred D | Pred A"]
    for i, st in enumerate(steps):
        h = shares_H[i]; d = shares_D[i]; a = shares_A[i]
        h = f"{h*100:.1f}%" if np.isfinite(h) else "-"
        d = f"{d*100:.1f}%" if np.isfinite(d) else "-"
        a = f"{a*100:.1f}%" if np.isfinite(a) else "-"
        lines.append(f"{st} | {h} | {d} | {a}")
    fig.text(0.02, 0.02, "\n".join(lines), fontsize=9, family='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_png = os.path.join(out_dir, 'one_figure_comparison.png')
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def plot_rps_acc(out_dir):
    csv_path = os.path.join(out_dir, 'comparison_summary.csv')
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    order = ['exp0','exp1','exp2','exp3']
    df['Step'] = pd.Categorical(df['Step'], categories=order, ordered=True)
    df = df.sort_values('Step')
    steps = df['Step'].tolist()
    x = np.arange(len(steps))
    rps = [ _safe_float(v) for v in df['RPS'].tolist() ]
    acc = [ _safe_float(v) for v in df['ACC'].tolist() ]

    fig = plt.figure(figsize=(10.5, 6.2), dpi=160)
    ax = fig.add_subplot(111)
    bars = ax.bar(x, rps, width=0.6, alpha=0.95, linewidth=1.2)
    hatches = ['/', '\\\\', 'x', '-', '+', 'o', 'O', '.', '*']
    for i, b in enumerate(bars):
        b.set_linewidth(1.2)
        b.set_hatch(hatches[i % len(hatches)] if len(steps) > 1 else '')
    for i, v in enumerate(rps):
        if np.isfinite(v): ax.text(x[i], v, f'{v:.6f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x, steps); ax.set_ylabel('RPS (lower is better)')
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(x, acc, marker='o', linewidth=2.3, label='ACC')
    ax2.set_ylim(0.0, 1.0); ax2.set_ylabel('ACC')
    ax.set_title('RPS & ACC across steps', fontsize=13)
    ax2.legend(loc='upper right', framealpha=0.9)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'rps_acc.png'), dpi=300); plt.close(fig)

def plot_f1_grouped(out_dir):
    sum_csv = os.path.join(out_dir, 'comparison_summary.csv')
    if not os.path.exists(sum_csv):
        return
    df_sum = pd.read_csv(sum_csv)
    order = ['exp0','exp1','exp2','exp3']
    df_sum['Step'] = pd.Categorical(df_sum['Step'], categories=order, ordered=True)
    df_sum = df_sum.sort_values('Step')
    steps = df_sum['Step'].tolist()

    F1H, F1D, F1A = [], [], []
    for st in steps:
        prf_fp = os.path.join(out_dir, f'prf_{st}.csv')
        if os.path.exists(prf_fp):
            prf = pd.read_csv(prf_fp, index_col=0)
            F1H.append(_safe_float(prf.loc['Home wins','F1-score']))
            F1D.append(_safe_float(prf.loc['Draws','F1-score']))
            F1A.append(_safe_float(prf.loc['Away wins','F1-score']))
        else:
            F1H.append(np.nan); F1D.append(np.nan); F1A.append(np.nan)

    x = np.arange(len(steps)); w = 0.22
    fig = plt.figure(figsize=(10.5, 6.2), dpi=160)
    ax = fig.add_subplot(111)
    b1 = ax.bar(x - w, F1H, width=w, alpha=0.95, linewidth=1.0)
    b2 = ax.bar(x,      F1D, width=w, alpha=0.95, linewidth=1.0)
    b3 = ax.bar(x + w,  F1A, width=w, alpha=0.95, linewidth=1.0)
    for bars in [b1, b2, b3]:
        for rect in bars:
            h = rect.get_height()
            if np.isfinite(h):
                ax.text(rect.get_x() + rect.get_width()/2, h, f'{h:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x, steps); ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('F1-score'); ax.set_title('Class-wise F1 across steps', fontsize=13)
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.legend(['F1 Home', 'F1 Draw', 'F1 Away'], loc='upper left', framealpha=0.9)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'f1_grouped.png'), dpi=300); plt.close(fig)

def plot_pred_distribution(out_dir):
    sum_csv = os.path.join(out_dir, 'comparison_summary.csv')
    if not os.path.exists(sum_csv):
        return
    df_sum = pd.read_csv(sum_csv)
    order = ['exp0','exp1','exp2','exp3']
    df_sum['Step'] = pd.Categorical(df_sum['Step'], categories=order, ordered=True)
    df_sum = df_sum.sort_values('Step')
    steps = df_sum['Step'].tolist()

    shares = []
    for st in steps:
        cm_fp = os.path.join(out_dir, f'confmat_{st}.csv')
        if os.path.exists(cm_fp):
            cm_df = pd.read_csv(cm_fp, index_col=0)
            col_sums = cm_df.sum(axis=0).values.astype(float)
            tot = col_sums.sum() if col_sums.sum()>0 else 1.0
            shares.append(tuple(col_sums / tot))
        else:
            shares.append((np.nan, np.nan, np.nan))
    H = [s[0] for s in shares]; D = [s[1] for s in shares]; A = [s[2] for s in shares]

    x = np.arange(len(steps))
    fig = plt.figure(figsize=(10.5, 5.2), dpi=160)
    ax = fig.add_subplot(111)
    bottom = np.zeros_like(x, dtype=float)
    for arr, lab in zip([H, D, A], ['H','D','A']):
        vals = [ (v if np.isfinite(v) else 0.0) for v in arr ]
        bars = ax.bar(x, vals, bottom=bottom, width=0.6, alpha=0.95, linewidth=1.0, label=lab)
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(x[i], bottom[i] + v/2, f'{v*100:.1f}%', ha='center', va='center', fontsize=8)
        bottom += np.array(vals)
    ax.set_xticks(x, steps); ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Predicted share'); ax.set_title('Prediction distribution (H/D/A) by step', fontsize=13)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.5)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'pred_distribution.png'), dpi=300); plt.close(fig)

def _line_compare_by_split(metric_dict_by_split, title, ylabel, save_path):
    import numpy as np
    import matplotlib.pyplot as plt
    steps_sorted = ['exp0','exp1','exp2','exp3']
    x = np.arange(len(steps_sorted))
    plt.figure(figsize=(9.8, 5.6), dpi=160)
    for split, d in metric_dict_by_split.items():
        ys = [d.get(s, np.nan) for s in steps_sorted]
        plt.plot(x, ys, marker='o', linewidth=2.0, label=split)
        for xi, yi in zip(x, ys):
            if np.isfinite(yi):
                plt.text(xi, yi, f'{float(yi):.4f}', ha='center', va='bottom', fontsize=9)
    plt.xticks(x, steps_sorted)
    plt.title(title, fontsize=13); plt.ylabel(ylabel, fontsize=11)
    plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)
    plt.legend(framealpha=0.9)
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()

def _radar_compare_f1_by_step(f1_by_step, title, save_path):
    import numpy as np
    import matplotlib.pyplot as plt
    labels = ['Home wins','Draws','Away wins']
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    plt.figure(figsize=(7,7), dpi=160)
    ax = plt.subplot(111, polar=True)
    for step, d in f1_by_step.items():
        vals = [d.get(l, np.nan) for l in labels]; vals += vals[:1]
        vals = [0.0 if (not np.isfinite(v)) else float(v) for v in vals]
        ax.plot(angles, vals, linewidth=2.0, marker='o', label=step)
        ax.fill(angles, vals, alpha=0.10)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(title, va='bottom')
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)
    ax.set_ylim(0.0, 1.0)
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.10))
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/mnt/train-decision-agent/dingzhijie/Trident-Cat code modification/数据/英超/赛季数据')
    parser.add_argument('--work_dir', default='/mnt/train-decision-agent/dingzhijie/Trident-Cat code modification/数据/英超')
    parser.add_argument('--out_dir',  default='/mnt/train-decision-agent/dingzhijie/Trident-Cat code modification/cat/英超/ablation')
    parser.add_argument('--steps', nargs='+', default=['exp0','exp1','exp2','exp3'],
                        choices=['exp0','exp1','exp2','exp3'])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Read data
    train_df, test_df = read_and_prepare(args.data_dir, args.work_dir)
    X_train_full = train_df[[c for c in FEATURE_COLS if c in train_df.columns]].values
    y_train_full = train_df['FTR'].values.astype(int)
    X_test_full  = test_df[[c for c in FEATURE_COLS if c in test_df.columns]].values
    y_test_full  = test_df['FTR'].values.astype(int)

    # Split masks
    test_src = test_df.get('src', pd.Series(['ALL']*len(test_df)))
    split_masks = {
        'df2223': (test_src == 'df2223'),
        'df2324': (test_src == 'df2324'),
        'ALL':    np.ones(len(test_src), dtype=bool)
    }
    rps_by_split = {k:{} for k in split_masks.keys()}
    acc_by_split = {k:{} for k in split_masks.keys()}
    f1_by_split  = {k:{} for k in split_masks.keys()}

    def eval_and_store_split(step_name, P_all, y_pred_all):
        for split, mask in split_masks.items():
            y_t = y_test_full[mask]
            P   = P_all[mask]
            y_p = y_pred_all[mask]
            rps = calc_rps(y_t, P, 3)
            acc = float((y_p == y_t).mean()) if len(y_t)>0 else float('nan')
            rps_by_split[split][step_name] = rps
            acc_by_split[split][step_name] = acc
            prec, rec, f1, _ = precision_recall_fscore_support(y_t, y_p, labels=[0,1,2], zero_division=0)
            f1_by_split[split][step_name] = {'Home wins':float(f1[0]), 'Draws':float(f1[1]), 'Away wins':float(f1[2])}

    # Containers
    probs_dict = {}
    rps_dict   = {}
    acc_dict   = {}
    summary_rows = []

    def save_conf_prf(y_true, y_pred, tag):
        labels = [0,1,2]; names = ['Home wins','Draws','Away wins']
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=names, columns=['Predicted win','Predicted draw','Predicted loss'])
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        prf_table = pd.DataFrame({'Precision':precision,'Recall':recall,'F1-score':f1}, index=names).applymap(lambda x: round(float(x),6))
        cm_df.to_csv(os.path.join(args.out_dir, f'confmat_{tag}.csv'), encoding='utf-8-sig')
        prf_table.to_csv(os.path.join(args.out_dir, f'prf_{tag}.csv'), encoding='utf-8-sig')

    # EXP0 — B365 baseline
    if 'exp0' in args.steps:
        P0 = odds_to_probs(test_df[['B365H','B365D','B365A']].values)
        probs_dict['exp0'] = P0
        y_pred0 = np.argmax(P0, axis=1)
        rps0 = calc_rps(y_test_full, P0, 3)
        acc0 = float((y_pred0 == y_test_full).mean())
        rps_dict['exp0'] = rps0; acc_dict['exp0'] = acc0
        pd.DataFrame(P0, columns=['P_H','P_D','P_A']).to_csv(os.path.join(args.out_dir,'probs_exp0.csv'), index=False, encoding='utf-8-sig')
        save_conf_prf(y_test_full, y_pred0, 'exp0')
        summary_rows.append(dict(Step='exp0', RPS=rps0, ACC=acc0, Note='B365 baseline (no training)'))
        eval_and_store_split('exp0', P0, y_pred0)
        print(f"[exp0] RPS={rps0:.12f} ACC={acc0:.6f}")

    # Train heads (only once per KD mode)
    heads_cache = {}

    def get_heads(use_kd: bool):
        key = 'kd' if use_kd else 'hard'
        if key in heads_cache:
            return heads_cache[key]
        models, splits, teachers = train_AB_heads(X_train_full, y_train_full, train_df, use_kd=use_kd)
        heads_cache[key] = (models, splits, teachers)
        return heads_cache[key]

    # EXP1 — hard labels; no cal/tb
    if 'exp1' in args.steps:
        (mA, mB), (X_fit, y_fit, X_cal, y_cal), _ = get_heads(use_kd=False)
        calib = calibrate_heads((mA, mB), X_cal, y_cal, do_calibration=False)
        P1 = assemble_prob_3class((mA, mB), calib, X_test_full, apply_tb=False)
        probs_dict['exp1'] = P1
        y_pred1 = np.argmax(P1, axis=1)
        rps1 = calc_rps(y_test_full, P1, 3)
        acc1 = float((y_pred1 == y_test_full).mean())
        rps_dict['exp1'] = rps1; acc_dict['exp1'] = acc1
        pd.DataFrame(P1, columns=['P_H','P_D','P_A']).to_csv(os.path.join(args.out_dir,'probs_exp1.csv'), index=False, encoding='utf-8-sig')
        save_conf_prf(y_test_full, y_pred1, 'exp1')
        summary_rows.append(dict(Step='exp1', RPS=rps1, ACC=acc1, Note='Two-head hard; no cal/tb/blend'))
        eval_and_store_split('exp1', P1, y_pred1)
        print(f"[exp1] RPS={rps1:.12f} ACC={acc1:.6f}")

    # EXP2 — +KD +calibration +t,b
    if 'exp2' in args.steps:
        (mA2, mB2), (X_fit2, y_fit2, X_cal2, y_cal2), _ = get_heads(use_kd=True)
        calib2 = calibrate_heads((mA2, mB2), X_cal2, y_cal2, do_calibration=True)
        P2 = assemble_prob_3class((mA2, mB2), calib2, X_test_full, apply_tb=True)
        probs_dict['exp2'] = P2
        y_pred2 = np.argmax(P2, axis=1)
        rps2 = calc_rps(y_test_full, P2, 3)
        acc2 = float((y_pred2 == y_test_full).mean())
        rps_dict['exp2'] = rps2; acc_dict['exp2'] = acc2
        pd.DataFrame(P2, columns=['P_H','P_D','P_A']).to_csv(os.path.join(args.out_dir,'probs_exp2.csv'), index=False, encoding='utf-8-sig')
        save_conf_prf(y_test_full, y_pred2, 'exp2')
        summary_rows.append(dict(Step='exp2', RPS=rps2, ACC=acc2, Note='+KD +calibration +t,b'))
        eval_and_store_split('exp2', P2, y_pred2)
        print(f"[exp2] RPS={rps2:.12f} ACC={acc2:.6f}")

    # EXP3 — + BPCF (Per-class Logit Blend + Platt Ensemble + Deviation Cap) + decision
    if 'exp3' in args.steps:
        if 'exp2' in probs_dict:
            P_model = probs_dict['exp2']
        else:
            (mA3, mB3), (X_fit3, y_fit3, X_cal3, y_cal3), _ = get_heads(use_kd=True)
            calib3 = calibrate_heads((mA3, mB3), X_cal3, y_cal3, do_calibration=True)
            P_model = assemble_prob_3class((mA3, mB3), calib3, X_test_full, apply_tb=True)
        B = probs_dict['exp0'] if 'exp0' in probs_dict else odds_to_probs(test_df[['B365H','B365D','B365A']].values)
        train_b365 = odds_to_probs(train_df[['B365H','B365D','B365A']].values)
        n_skip = int(len(train_df) * 2 / 8)
        Pf = apply_bpcf(B=B, M=P_model, train_b365=train_b365, y_train=y_train_full, n_skip=n_skip)
        P_dec = classwise_temp(Pf, DECISION['alphaH'], DECISION['alphaD'], DECISION['alphaA'])
        y_pred3 = predict_with_params(P_dec, DECISION['tauH'], DECISION['tauD'], DECISION['tauA'], DECISION['gamma'])
        probs_dict['exp3'] = Pf
        rps3 = calc_rps(y_test_full, Pf, 3)
        acc3 = float((y_pred3 == y_test_full).mean())
        rps_dict['exp3'] = rps3; acc_dict['exp3'] = acc3
        pd.DataFrame(Pf, columns=['P_H','P_D','P_A']).to_csv(os.path.join(args.out_dir,'probs_exp3.csv'), index=False, encoding='utf-8-sig')
        save_conf_prf(y_test_full, y_pred3, 'exp3')
        summary_rows.append(dict(Step='exp3', RPS=rps3, ACC=acc3, Note='+BPCF + decision'))
        eval_and_store_split('exp3', Pf, y_pred3)
        print(f"[exp3] RPS={rps3:.12f} ACC={acc3:.6f}")

    # Summaries
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        order = ['exp0','exp1','exp2','exp3']
        df_sum['Step'] = pd.Categorical(df_sum['Step'], categories=order, ordered=True)
        df_sum = df_sum.sort_values('Step')
        df_sum.to_csv(os.path.join(args.out_dir, 'comparison_summary.csv'), index=False, encoding='utf-8-sig')
        print("\n=== Final Comparison (exp0..exp3) ===")
        print(df_sum.to_string(index=False))

    # One-figure overview
    one_figure_plot(args.out_dir)
    # Category plots
    plot_rps_acc(args.out_dir)
    plot_f1_grouped(args.out_dir)
    plot_pred_distribution(args.out_dir)

    # Split-wise CSV + plots
    rows_split = []
    for split in ['df2223','df2324','ALL']:
        if split in rps_by_split:
            for step, rpsv in rps_by_split[split].items():
                accv = acc_by_split[split].get(step, float('nan'))
                f1s = f1_by_split[split].get(step, {})
                rows_split.append(dict(Split=split, Step=step, RPS=rpsv, ACC=accv,
                                       F1_H=f1s.get('Home wins', np.nan),
                                       F1_D=f1s.get('Draws', np.nan),
                                       F1_A=f1s.get('Away wins', np.nan)))
    if rows_split:
        df_split = pd.DataFrame(rows_split)
        df_split.to_csv(os.path.join(args.out_dir, 'comparison_summary_by_split.csv'), index=False, encoding='utf-8-sig')
        # RPS
        rps_plot_map = {}
        for split in df_split['Split'].unique():
            sub = df_split[df_split['Split']==split].copy()
            sub['Step'] = pd.Categorical(sub['Step'], categories=['exp0','exp1','exp2','exp3'], ordered=True)
            sub = sub.sort_values('Step')
            rps_plot_map[split] = {row['Step']: float(row['RPS']) for _, row in sub.iterrows()}
        _line_compare_by_split(rps_plot_map, 'RPS by split (lower is better)', 'RPS',
                               os.path.join(args.out_dir, 'rps_by_split.png'))
        # ACC
        acc_plot_map = {}
        for split in df_split['Split'].unique():
            sub = df_split[df_split['Split']==split].copy()
            sub['Step'] = pd.Categorical(sub['Step'], categories=['exp0','exp1','exp2','exp3'], ordered=True)
            sub = sub.sort_values('Step')
            acc_plot_map[split] = {row['Step']: float(row['ACC']) for _, row in sub.iterrows()}
        _line_compare_by_split(acc_plot_map, 'Accuracy by split', 'Accuracy',
                               os.path.join(args.out_dir, 'acc_by_split.png'))
        # Radar
        for split in df_split['Split'].unique():
            sub = df_split[df_split['Split']==split].copy()
            sub['Step'] = pd.Categorical(sub['Step'], categories=['exp0','exp1','exp2','exp3'], ordered=True)
            sub = sub.sort_values('Step')
            f1_map = {}
            for _, row in sub.iterrows():
                f1_map[row['Step']] = {'Home wins': _safe_float(row['F1_H']),
                                       'Draws':     _safe_float(row['F1_D']),
                                       'Away wins': _safe_float(row['F1_A'])}
            _radar_compare_f1_by_step(f1_map, f'F1 Radar – {split}',
                                      os.path.join(args.out_dir, f'f1_radar_{split}.png'))

    print("\nAll outputs saved to:", args.out_dir)

if __name__ == '__main__':
    main()
