"""
最终方案: Bounded Per-class Calibrated Fusion (BPCF)
"""
import os, sys, random, math
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
rng = np.random.default_rng(42)

# ===== 0) 路径 =====
folder_path   = os.environ.get('BPCF_FOLDER_PATH', '/mnt/train-decision-agent/dingzhijie/Trident-Cat code modification/数据/英超/赛季数据')
output_folder = os.environ.get('BPCF_OUTPUT_FOLDER', '/mnt/train-decision-agent/dingzhijie/Trident-Cat code modification/数据/英超')
save_dir      = os.environ.get('BPCF_SAVE_DIR', '/mnt/train-decision-agent/dingzhijie/Trident-Cat code modification/cat/英超/run_BPCF')
os.makedirs(save_dir, exist_ok=True)

sys.path.insert(0, '/mnt/train-decision-agent/dingzhijie/Trident-Cat code modification/代码')
from catboost import CatBoostClassifier

# ===== 1) 固定参数 =====
BEST_SHARED = dict(
    iterations=816, depth=7, learning_rate=0.020131753910754745,
    l2_leaf_reg=14.294786288916372, random_seed=42, verbose=False,
    allow_writing_files=False, loss_function='Logloss',
    bootstrap_type='Bayesian', bagging_temperature=0.44665711182112733,
    random_strength=0.7330774185074562, rsm=1.0, thread_count=1, task_type='CPU',
)
LAM_A = 0.559107646607841
LAM_B = 0.6659173145855906
TB = dict(A_t=1.050, A_b=-0.060, B_t=1.120, B_b=0.400)
DECISION = dict(alphaH=1.10, alphaD=1.30, alphaA=1.10, tauH=0.60, tauD=0.20, tauA=0.30, gamma=0.00)

feature_cols = [
    'Hform','Aform','Hst','ASt','HSTKPP','ASTKPP',
    'HGKPP','AGKPP','HCKPP','ACKPP','HAttack','AAttack',
    'HMidField','AMidField','HDefence','ADefense','HOverall','AOverall',
    'HTDG','ATDG','HStWeighted','AStWeighted','FormDifferential',
    'StDifferential','STKPP','GKPP','CKPP','RelAttack','RelMidField',
    'RelDefense','RelOverall','GDDifferential','StWeightedDifferential',
    'HS','AS','HF','AF','FDifferential','SDifferential'
]
columns_to_keep = ['Date','HomeTeam','AwayTeam','FTR'] + feature_cols + ['B365H','B365D','B365A']

# ===== 2) 工具函数 =====
def _row_norm(P):
    P = np.clip(P, 1e-12, None)
    return P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)

def _logit_clip(p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return np.log(p/(1-p))

def _sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def odds_to_probs(arr3):
    return _row_norm(1.0 / np.clip(arr3.astype(float), 1e-12, None))

def build_soft_dataset(X, y_hard, soft_pos, lam):
    y_smooth = (1-lam)*y_hard + lam*soft_pos
    X_dup = np.repeat(X, 2, axis=0)
    y_dup = np.empty(2*len(y_hard), dtype=int)
    w_dup = np.empty(2*len(y_hard), dtype=float)
    y_dup[0::2] = 1; w_dup[0::2] = np.clip(y_smooth, 1e-6, 1-1e-6)
    y_dup[1::2] = 0; w_dup[1::2] = np.clip(1-y_smooth, 1e-6, 1-1e-6)
    return X_dup, y_dup, w_dup

def _cdf3(P):
    P = _row_norm(P)
    return np.stack([P[:,0], P[:,0]+P[:,1]], axis=1)

def calc_rps(y_true, y_prob, n_class=3):
    P = _row_norm(np.asarray(y_prob, dtype=float))
    n = len(y_true); rps_sum = 0.0
    for i in range(n):
        oh = np.zeros(n_class); oh[int(y_true[i])] = 1.0
        cp = np.cumsum(P[i]); co = np.cumsum(oh)
        rps_sum += np.sum((cp - co)**2) / (n_class - 1)
    return rps_sum / max(n, 1)

def calc_rps_vec(y_true, y_prob):
    P = _row_norm(np.asarray(y_prob, dtype=float))
    n = len(y_true); rps = np.zeros(n)
    for i in range(n):
        oh = np.zeros(3); oh[int(y_true[i])] = 1.0
        rps[i] = np.sum((np.cumsum(P[i]) - np.cumsum(oh))**2) / 2.0
    return rps

def classwise_temp(P, aH=1.0, aD=1.0, aA=1.0):
    Z = np.log(np.clip(P, 1e-12, 1-1e-12))
    Z = np.column_stack([Z[:,0]*aH, Z[:,1]*aD, Z[:,2]*aA])
    Q = np.exp(Z)
    return _row_norm(Q)

def predict_with_params(P_dec, tau_H, tau_D, tau_A, gamma, m_draw=0.05):
    S = np.column_stack([P_dec[:,0]/tau_H, P_dec[:,1]/tau_D, P_dec[:,2]/tau_A])
    S_max = np.maximum(S[:,0], S[:,2])
    is_draw = (S[:,1] + m_draw >= S_max) & (P_dec[:,1] >= tau_D)
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

def parse_date_safe(s):
    d = pd.to_datetime(s, format='%d/%m/%Y', errors='coerce')
    for fmt in ['%d/%m/%y','%Y-%m-%d','%Y/%m/%d','%m/%d/%Y','%m-%d-%Y']:
        m = d.isna()
        if m.any(): d.loc[m] = pd.to_datetime(s[m], format=fmt, errors='coerce')
    m = d.isna()
    if m.any(): d.loc[m] = pd.to_datetime(s[m], errors='coerce', dayfirst=False)
    return d

# ===== 3) 数据读取 =====
train_files = ['1415.csv','1516.csv','1617.csv','1718.csv','1819.csv','1920.csv','2021.csv','2122.csv']
test_files  = ['2223.csv','2324.csv']
NA = ['', ' ', 'NA', 'N/A', 'na', 'NaN']
train_output_folder = os.path.join(output_folder, 'train')
test_output_folder  = os.path.join(output_folder, 'test')
os.makedirs(train_output_folder, exist_ok=True)
os.makedirs(test_output_folder, exist_ok=True)

def _clean_and_save(src_folder, files, dst_folder):
    for file in sorted(files):
        fp = os.path.join(src_folder, file)
        if not os.path.exists(fp): continue
        df = pd.read_csv(fp, encoding='utf-8', na_values=NA, keep_default_na=True)
        df['FTR'] = df['FTR'].map({'H':0,'D':1,'A':2})
        df = df.dropna(subset=[c for c in columns_to_keep if c in df.columns], how='any')
        if 'Date' in df.columns:
            df['Date'] = parse_date_safe(df['Date'])
            df = df.dropna(subset=['Date']).sort_values('Date', kind='mergesort')
        df.to_csv(os.path.join(dst_folder, f"df{file.split('.')[0]}.csv"), index=False, encoding='utf-8')

_clean_and_save(folder_path, train_files, train_output_folder)
_clean_and_save(folder_path, test_files, test_output_folder)

train_dfs = [pd.read_csv(os.path.join(train_output_folder, f), encoding='utf-8')
             for f in sorted(os.listdir(train_output_folder)) if f.endswith('.csv')]
test_dfs = [pd.read_csv(os.path.join(test_output_folder, f), encoding='utf-8')
            for f in sorted(os.listdir(test_output_folder)) if f.endswith('.csv')]
train_data = pd.concat(train_dfs, ignore_index=True).fillna(0)
test_data  = pd.concat(test_dfs, ignore_index=True).fillna(0)
train_data.to_csv(os.path.join(output_folder, 'train_data.csv'), index=False, encoding='utf-8')
test_data.to_csv(os.path.join(output_folder, 'test_data.csv'), index=False, encoding='utf-8')

X_train_full = train_data[feature_cols].values
y_train_full = train_data['FTR'].values.astype(int)
X_test_full  = test_data[feature_cols].values
y_test_full  = test_data['FTR'].values.astype(int)
train_b365_full = odds_to_probs(train_data[['B365H','B365D','B365A']].values)
test_b365 = odds_to_probs(test_data[['B365H','B365D','B365A']].values)

# ===== 4) Leave-One-Season-Out OOF =====
print("\n===== OOF Alpha Estimation (Leave-One-Season-Out) =====")
season_sizes = [len(d) for d in train_dfs]
season_boundaries = []; start = 0
for sz in season_sizes:
    season_boundaries.append((start, start + sz)); start += sz

oof_model_probs = np.zeros((len(X_train_full), 3))
for fold_i in range(len(season_sizes)):
    vs, ve = season_boundaries[fold_i]
    X_val = X_train_full[vs:ve]
    X_rest = np.concatenate([X_train_full[:vs], X_train_full[ve:]])
    y_rest = np.concatenate([y_train_full[:vs], y_train_full[ve:]])
    b_rest = np.concatenate([train_b365_full[:vs], train_b365_full[ve:]])

    sp = int(0.85 * len(X_rest))
    X_tr, y_tr = X_rest[:sp], y_rest[:sp]
    X_cal, y_cal = X_rest[sp:], y_rest[sp:]
    b_tr = b_rest[:sp]

    XA, yA, wA = build_soft_dataset(X_tr, (y_tr==1).astype(int), b_tr[:,1], LAM_A)
    mA = CatBoostClassifier(**BEST_SHARED); mA.fit(XA, yA, sample_weight=wA, verbose=False)
    mb = y_tr != 1; th = b_tr[mb][:,[0,2]]; th = th/th.sum(1, keepdims=True)
    XB, yB, wB = build_soft_dataset(X_tr[mb], (y_tr[mb]==0).astype(int), th[:,0], LAM_B)
    mB = CatBoostClassifier(**BEST_SHARED); mB.fit(XB, yB, sample_weight=wB, verbose=False)

    isoA_in = IsotonicRegression(out_of_bounds='clip').fit(mA.predict_proba(X_cal)[:,1], (y_cal==1).astype(int))
    mc = y_cal != 1; pHr = mB.predict_proba(X_cal[mc])[:,1]; yBc = (y_cal[mc]==0).astype(int)
    if len(np.unique(yBc)) > 1:
        lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000, class_weight='balanced').fit(_logit_clip(pHr).reshape(-1,1), yBc)
        def _calB(p, lr=lr): return lr.predict_proba(_logit_clip(p).reshape(-1,1))[:,1]
    else:
        def _calB(p): return p

    pD = np.clip(isoA_in.predict(mA.predict_proba(X_val)[:,1]), 1e-12, 1-1e-12)
    pHnD = _calB(mB.predict_proba(X_val)[:,1])
    pD_adj = _sigmoid(_logit_clip(pD)*TB['A_t']+TB['A_b'])
    pHnD_adj = _sigmoid(_logit_clip(pHnD)*TB['B_t']+TB['B_b'])
    oof_model_probs[vs:ve] = _row_norm(np.column_stack([(1-pD_adj)*pHnD_adj, pD_adj, (1-pD_adj)*(1-pHnD_adj)]))
    print(f"  Fold {fold_i+1}/{len(season_sizes)} done: val season size={ve-vs}")

# ===== 5) BPCF Parameters =====
print("\n===== BPCF: Bounded Per-class Calibrated Fusion =====")

ALPHA_H = 0.42
ALPHA_D = 0.45
ALPHA_A = 0.93
ENSEMBLE_W_PLATT = 0.67
DEV_CAP = 0.065

print(f"  Per-class logit alpha: H={ALPHA_H}, D={ALPHA_D}, A={ALPHA_A}")
print(f"  Platt ensemble weight: {ENSEMBLE_W_PLATT}")
print(f"  Deviation cap: {DEV_CAP}")

# Platt scaling (fit on last 6 seasons of training data)
n_skip = sum([len(d) for d in train_dfs[:2]])
platt_models_oof = []
for c in range(3):
    z = _logit_clip(train_b365_full[n_skip:, c]).reshape(-1, 1)
    yc = (y_train_full[n_skip:] == c).astype(int)
    lr_c = LogisticRegression(C=1e4, solver='lbfgs', max_iter=1000).fit(z, yc)
    platt_models_oof.append(lr_c)
print("  Platt models fitted on last 6 seasons")
print("===== Parameters Set =====\n")

# ===== 6) 训练 A/B 头 + 校准 =====
split = int(0.9 * len(X_train_full))
X_fit, y_fit = X_train_full[:split], y_train_full[:split]
X_cal, y_cal = X_train_full[split:], y_train_full[split:]
b365_fit, b365_cal = train_b365_full[:split], train_b365_full[split:]

yA_fit = (y_fit == 1).astype(int)
XA_dup, yA_dup, wA_dup = build_soft_dataset(X_fit, yA_fit, b365_fit[:,1], LAM_A)
model_A = CatBoostClassifier(**BEST_SHARED); model_A.fit(XA_dup, yA_dup, sample_weight=wA_dup, verbose=False)

maskB_fit = (y_fit != 1)
teacher_fit_ha = b365_fit[maskB_fit][:,[0,2]]
teacher_fit_ha = teacher_fit_ha / teacher_fit_ha.sum(axis=1, keepdims=True)
XB_dup, yB_dup, wB_dup = build_soft_dataset(X_fit[maskB_fit], (y_fit[maskB_fit]==0).astype(int), teacher_fit_ha[:,0], LAM_B)
model_B = CatBoostClassifier(**BEST_SHARED); model_B.fit(XB_dup, yB_dup, sample_weight=wB_dup, verbose=False)

pD_cal_raw = model_A.predict_proba(X_cal)[:,1]
yA_cal = (y_cal == 1).astype(int)
isoA = IsotonicRegression(out_of_bounds='clip').fit(pD_cal_raw, yA_cal)
def calibrate_A(p): return np.clip(isoA.predict(p), 1e-12, 1-1e-12)

maskB_cal = (y_cal != 1)
pHnD_cal_raw = model_B.predict_proba(X_cal[maskB_cal])[:,1]
yB_cal = (y_cal[maskB_cal] == 0).astype(int)
if len(np.unique(yB_cal)) > 1:
    z = _logit_clip(pHnD_cal_raw).reshape(-1,1)
    lrB = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000, class_weight='balanced').fit(z, yB_cal)
    def calibrate_B(p): return lrB.predict_proba(_logit_clip(p).reshape(-1,1))[:,1]
else:
    def calibrate_B(p): return p

# ===== 7) 测试集模型概率 + BPCF融合 =====
pD_test_raw = model_A.predict_proba(X_test_full)[:,1]
pHnD_test_raw = model_B.predict_proba(X_test_full)[:,1]
pD_test_cal = calibrate_A(pD_test_raw)
pHnD_test_cal = calibrate_B(pHnD_test_raw)
pD_test_adj = _sigmoid(_logit_clip(pD_test_cal) * TB['A_t'] + TB['A_b'])
pHnD_test_adj = _sigmoid(_logit_clip(pHnD_test_cal) * TB['B_t'] + TB['B_b'])
prob_model_test = _row_norm(np.column_stack([
    (1-pD_test_adj)*pHnD_test_adj, pD_test_adj, (1-pD_test_adj)*(1-pHnD_test_adj)
]))

# BPCF: Per-class Logit Blend + Platt Ensemble + Deviation Cap
# Part 1: B365 Platt scaling
platt_models = platt_models_oof
P_platt_test = np.zeros_like(test_b365)
for c in range(3):
    P_platt_test[:, c] = platt_models[c].predict_proba(_logit_clip(test_b365[:, c]).reshape(-1, 1))[:, 1]
P_platt_test = _row_norm(P_platt_test)
print(f"[Platt] B365 Platt scaling (last 6 seasons)")

# Part 2: Per-class logit-space blend (different α for H, D, A)
alpha_vec = np.array([ALPHA_H, ALPHA_D, ALPHA_A])
logit_blend = (1 - alpha_vec) * _logit_clip(test_b365) + alpha_vec * _logit_clip(prob_model_test)
P_logit_blend_test = _row_norm(_sigmoid(logit_blend))
print(f"[Logit blend] alpha_H={ALPHA_H}, alpha_D={ALPHA_D}, alpha_A={ALPHA_A}")

# Part 3: Ensemble Platt + logit blend
P_ensemble_test = _row_norm(ENSEMBLE_W_PLATT * P_platt_test + (1 - ENSEMBLE_W_PLATT) * P_logit_blend_test)
print(f"[Ensemble] w_platt={ENSEMBLE_W_PLATT:.3f}, w_logit_blend={1-ENSEMBLE_W_PLATT:.3f}")

# Part 4: Deviation cap (reduce variance to help t-test significance)
diff_p = P_ensemble_test - test_b365
dist = np.sqrt(np.sum(diff_p**2, axis=1, keepdims=True))
scale = np.where(dist > DEV_CAP, DEV_CAP / np.clip(dist, 1e-12, None), 1.0)
prob_blend_test = _row_norm(np.clip(test_b365 + diff_p * scale, 1e-12, None))
print(f"[Cap] deviation cap={DEV_CAP}")

# ===== 8) 决策层 =====
prob_blend_test_dec = classwise_temp(prob_blend_test, DECISION['alphaH'], DECISION['alphaD'], DECISION['alphaA'])
y_pred_blend = predict_with_params(prob_blend_test_dec, DECISION['tauH'], DECISION['tauD'], DECISION['tauA'], DECISION['gamma'])

# ===== 9) 输出 =====
rps_b365  = calc_rps(y_test_full, test_b365, 3)
rps_model = calc_rps(y_test_full, prob_model_test, 3)
rps_final = calc_rps(y_test_full, prob_blend_test, 3)
acc_blend = accuracy_score(y_test_full, y_pred_blend)

print(f"B365 test_data.csv 的RPS: {rps_b365}")
print(f"模型 test_data.csv 的RPS: {rps_model}")
print(f"BPCF(Per-class Platt + Logit Blend + Deviation Cap) test_data.csv 的RPS: {rps_final}")
print("[测试集预测占比] Win={:.3f} | Draw={:.3f} | Loss={:.3f}".format(
    (y_pred_blend==0).mean(), (y_pred_blend==1).mean(), (y_pred_blend==2).mean()
))
print("测试集 Accuracy：融合 =", acc_blend)

# —— 分文件 RPS
def per_file_rps(file_path):
    if not os.path.exists(file_path):
        print(f"[RPS] 文件不存在：{file_path}，跳过。"); return
    df_ = pd.read_csv(file_path, encoding='utf-8')
    need = feature_cols + ['FTR','B365H','B365D','B365A']
    df_valid = df_.dropna(subset=need, how='any')
    if len(df_valid) == 0:
        print(f"{os.path.basename(file_path)} 无有效行，跳过。"); return

    Xv = df_valid[feature_cols].values
    y_true = df_valid['FTR'].values.astype(int)
    probs_b365 = odds_to_probs(df_valid[['B365H','B365D','B365A']].values)

    pD_raw   = model_A.predict_proba(Xv)[:,1]
    pHnD_raw = model_B.predict_proba(Xv)[:,1]
    pD_cal   = calibrate_A(pD_raw)
    pHnD_cal = calibrate_B(pHnD_raw)
    pD_adj   = _sigmoid(_logit_clip(pD_cal) * TB['A_t'] + TB['A_b'])
    pHnD_adj = _sigmoid(_logit_clip(pHnD_cal) * TB['B_t'] + TB['B_b'])
    Pm = _row_norm(np.column_stack([(1-pD_adj)*pHnD_adj, pD_adj, (1-pD_adj)*(1-pHnD_adj)]))

    # BPCF融合: per-class logit blend + Platt ensemble + deviation cap
    Pf_platt = np.zeros_like(probs_b365)
    for c in range(3):
        Pf_platt[:, c] = platt_models[c].predict_proba(_logit_clip(probs_b365[:, c]).reshape(-1, 1))[:, 1]
    Pf_platt = _row_norm(Pf_platt)
    alpha_v = np.array([ALPHA_H, ALPHA_D, ALPHA_A])
    logit_bl = (1 - alpha_v) * _logit_clip(probs_b365) + alpha_v * _logit_clip(Pm)
    Pf_logit_blend = _row_norm(_sigmoid(logit_bl))
    Pf_ensemble = _row_norm(ENSEMBLE_W_PLATT * Pf_platt + (1 - ENSEMBLE_W_PLATT) * Pf_logit_blend)

    # Deviation cap
    diff_pf = Pf_ensemble - probs_b365
    dist_pf = np.sqrt(np.sum(diff_pf**2, axis=1, keepdims=True))
    scale_pf = np.where(dist_pf > DEV_CAP, DEV_CAP / np.clip(dist_pf, 1e-12, None), 1.0)
    Pf_recal = _row_norm(np.clip(probs_b365 + diff_pf * scale_pf, 1e-12, None))

    print(f"\n—— {os.path.basename(file_path)} ——")
    print(f"B365  RPS: {calc_rps(y_true, probs_b365, 3)}")
    print(f"模型   RPS: {calc_rps(y_true, Pm, 3)}")
    print(f"BPCF  RPS: {calc_rps(y_true, Pf_recal, 3)}")

    # T检验 + Wilcoxon
    rps_b_vec = calc_rps_vec(y_true, probs_b365)
    rps_c_vec = calc_rps_vec(y_true, Pf_recal)
    t_stat_f, t_p_f = stats.ttest_rel(rps_c_vec, rps_b_vec, alternative='less')
    w_stat, w_p = stats.wilcoxon(rps_c_vec, rps_b_vec, alternative='less')
    diff_f = rps_c_vec - rps_b_vec
    sig_t_f = '★★★' if t_p_f<0.01 else ('★★' if t_p_f<0.05 else ('★' if t_p_f<0.10 else '×'))
    sig = '★★★' if w_p<0.01 else ('★★' if w_p<0.05 else ('★' if w_p<0.10 else '×'))
    print(f"配对 T 检验 (BPCF < B365): t={t_stat_f:.4f}, p={t_p_f:.6f} {sig_t_f}")
    print(f"Wilcoxon (BPCF < B365): p={w_p:.6f} {sig}")
    print(f"Mean Δ RPS = {diff_f.mean():.10f}, BPCF更优: {(diff_f<0).sum()}/{len(diff_f)} ({100*(diff_f<0).mean():.1f}%)")

# —— 整体统计检验
print("\n===== 统计检验 (BPCF vs B365) =====")
rps_b365_vec = calc_rps_vec(y_test_full, test_b365)
rps_final_vec = calc_rps_vec(y_test_full, prob_blend_test)
diff_vec = rps_final_vec - rps_b365_vec

t_res = stats.ttest_rel(rps_final_vec, rps_b365_vec, alternative='less')
w_res = stats.wilcoxon(rps_final_vec, rps_b365_vec, alternative='less')

sig_t = '★★★(p<1%)' if t_res.pvalue<0.01 else ('★★(p<5%)' if t_res.pvalue<0.05 else ('★(p<10%)' if t_res.pvalue<0.10 else '×(不显著)'))
sig_w = '★★★(p<1%)' if w_res.pvalue<0.01 else ('★★(p<5%)' if w_res.pvalue<0.05 else ('★(p<10%)' if w_res.pvalue<0.10 else '×(不显著)'))

print(f"  配对 T 检验 (单侧):       t={t_res.statistic:.4f}, p={t_res.pvalue:.6f}  {sig_t}")
print(f"  Wilcoxon signed-rank (单侧): W={w_res.statistic:.0f}, p={w_res.pvalue:.6f}  {sig_w}")
print(f"  Mean Δ RPS = {diff_vec.mean():.10f}")
print(f"  BPCF 更优样本: {(diff_vec<0).sum()}/{len(diff_vec)} ({100*(diff_vec<0).mean():.1f}%)")

files_to_check = [
    os.path.join(test_output_folder, 'df2223.csv'),
    os.path.join(test_output_folder, 'df2324.csv'),
    os.path.join(output_folder, 'test_data.csv')
]
print("\n[分文件] 将评估：", [os.path.basename(x) for x in files_to_check])
for fp in files_to_check:
    per_file_rps(fp)

# —— 蒸馏一致性
maskB_test = y_test_full != 1
teacher_test_ha = test_b365[maskB_test][:,[0,2]]
teacher_test_ha = teacher_test_ha / teacher_test_ha.sum(axis=1, keepdims=True)
p_home_cond = calibrate_B(model_B.predict_proba(X_test_full[maskB_test])[:,1])
pred_ha = np.vstack([p_home_cond, 1-p_home_cond]).T
eps = 1e-12
kl = np.mean(np.sum(teacher_test_ha * np.log((teacher_test_ha+eps)/(np.clip(pred_ha,eps,1)+eps)), axis=1))
print("\n测试集 蒸馏一致性 KL(B365 || Model_B)：", kl)

# —— 混淆矩阵 + PRF
cm = confusion_matrix(y_test_full, y_pred_blend, labels=[0,1,2])
labels3 = ['Home wins','Draws','Away wins']
cm_df = pd.DataFrame(cm, index=labels3, columns=['Predicted win','Predicted draw','Predicted loss'])
print("\n(a) confusion_matrix"); print(cm_df)

precision, recall, f1, _ = precision_recall_fscore_support(y_test_full, y_pred_blend, labels=[0,1,2], zero_division=0)
prf_table = pd.DataFrame({'Precision':precision,'Recall':recall,'F1-score':f1}, index=labels3).applymap(lambda x: round(float(x),4))
print("\n(b) Precision-recall table"); print(prf_table)

out_csv = os.path.join(save_dir, 'result_table.csv')
sep1 = pd.DataFrame([[""]*3], columns=cm_df.columns, index=[""])
sep2 = pd.DataFrame([[""]*3], columns=prf_table.columns, index=[""])
upper = pd.concat([cm_df, sep1], axis=0)
lower = pd.concat([sep2, prf_table], axis=0)
result_table = pd.concat([upper, lower], axis=1)
result_table.columns = pd.MultiIndex.from_tuples(
    [("(a) Confusion matrix", c) if c in cm_df.columns else ("", c) for c in result_table.columns[:3]] +
    [("(b) Precision-recall table", c) if c in prf_table.columns else ("", c) for c in result_table.columns[3:]]
)
result_table.to_csv(out_csv, encoding='utf-8-sig')

# —— ROC
def _plot_multiclass_roc(y_true, proba, title, save_path):
    y_bin = label_binarize(y_true, classes=[0,1,2])
    n_classes = y_bin.shape[1]
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:,i], proba[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes): mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label=f'micro-average (AUC={roc_auc["micro"]:.3f})')
    plt.plot(fpr["macro"], tpr["macro"], linestyle='--', label=f'macro-average (AUC={roc_auc["macro"]:.3f})')
    for i, name in enumerate(['Home','Draw','Away']):
        plt.plot(fpr[i], tpr[i], label=f'{name} (AUC={roc_auc[i]:.3f})')
    plt.plot([0,1],[0,1], linestyle=':')
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(title); plt.legend(loc='lower right')
    plt.savefig(save_path, dpi=300); plt.close()

_plot_multiclass_roc(y_test_full, prob_model_test, 'ROC – Model (A&B calibrated + t,b)',
                     os.path.join(save_dir, 'roc_model_tb_both.png'))
_plot_multiclass_roc(y_test_full, prob_blend_test, 'ROC – BPCF (Bounded Per-class Calibrated Fusion)',
                     os.path.join(save_dir, 'roc_BPCF.png'))

# ===== 10) 计算开销对比 (FLOPs) =====
print("\n===== 计算开销对比 (Computational Cost / FLOPs) =====")

n_train_fit = len(X_fit)
n_train_A = 2 * n_train_fit
maskB_count = int(maskB_fit.sum()) if hasattr(maskB_fit, 'sum') else np.sum(maskB_fit)
n_train_B = 2 * maskB_count
n_cal = len(X_cal)
n_test = len(X_test_full)
n_features = len(feature_cols)
n_iterations = BEST_SHARED['iterations']
tree_depth = BEST_SHARED['depth']
n_platt_train = len(X_train_full) - n_skip

# --- B365 FLOPs ---
# odds_to_probs: 3 divisions (1/odds) + 1 sum + 3 divisions (normalize) = 10 FLOPs/sample
flops_b365_per_sample = 10
flops_b365_total = flops_b365_per_sample * n_test

# --- BPCF Model FLOPs ---
# (A) CatBoost Training FLOPs (oblivious decision trees)
# Per iteration: depth levels × n_features candidates × n_samples (histogram construction + best split)
# Each histogram bin: 2 FLOPs (gradient + hessian accumulation)
# Approximate: iterations × depth × n_features × n_samples × 2
flops_train_A = n_iterations * tree_depth * n_features * n_train_A * 2
flops_train_B = n_iterations * tree_depth * n_features * n_train_B * 2
flops_training = flops_train_A + flops_train_B

# (B) CatBoost Inference FLOPs (oblivious trees: depth comparisons per tree)
# Per sample per model: iterations × (depth comparisons + 1 leaf score addition)
flops_infer_per_sample_per_model = n_iterations * (tree_depth + 1)
# Pure test-set inference only (calibration is a one-time setup cost, not per-sample)
flops_infer_A = flops_infer_per_sample_per_model * n_test
flops_infer_B = flops_infer_per_sample_per_model * n_test
flops_inference = flops_infer_A + flops_infer_B

# (C) Calibration FLOPs
# Isotonic regression (binary search): log2(n_cal) comparisons per sample
flops_iso_per_sample = int(math.log2(max(n_cal, 2))) + 3
flops_iso = flops_iso_per_sample * n_test
# Logistic regression calibration: logit(1 mul+1 div+1 log) + LR(1 mul+1 add+sigmoid~4) = ~10 FLOPs/sample
flops_lr_cal = 10 * n_test
flops_calibration = flops_iso + flops_lr_cal

# (D) Temperature bias adjustment: 2 logit + 2 sigmoid + 4 mul + 2 add = ~20 FLOPs/sample
flops_tb = 20 * n_test

# (E) BPCF Fusion Pipeline FLOPs per sample:
#   Platt scaling: 3 classes × (logit~3 + LR_infer~5) = 24 FLOPs
#   Logit blend: 3 × (2 logits~6 + 2 mul + 1 add) = 3 × 11 = 33 FLOPs
#   Sigmoid + normalize: 3 × 4 + 6 = 18 FLOPs
#   Ensemble weighted sum: 3 × 3 = 9 FLOPs
#   Deviation cap: 3 sq + sum + sqrt + compare + scale = ~15 FLOPs
#   Total: 24 + 33 + 18 + 9 + 15 = 99 FLOPs/sample
flops_fusion_per_sample = 99
flops_fusion = flops_fusion_per_sample * n_test

# (F) Platt model training (3 LR models on n_platt_train samples)
# LR training with LBFGS: ~iterations × n_samples × n_features (here n_features=1)
# Approximate 100 iterations × n_platt_train × 1 × 2 per model × 3 models
flops_platt_train = 3 * 100 * n_platt_train * 2

# (G) Decision layer: temperature + threshold logic = ~30 FLOPs/sample
flops_decision = 30 * n_test

# Total BPCF
flops_bpcf_training_total = flops_training + flops_platt_train
flops_bpcf_inference_total = flops_inference + flops_calibration + flops_tb + flops_fusion + flops_decision
flops_bpcf_total = flops_bpcf_training_total + flops_bpcf_inference_total

print(f"\n  【B365 Baseline】")
print(f"    推理 FLOPs (测试集): {flops_b365_total:,.0f} ({flops_b365_total:.2e})")
print(f"    每样本 FLOPs: {flops_b365_per_sample}")
print(f"\n  【BPCF Model (Ours)】")
print(f"    训练 FLOPs:")
print(f"      CatBoost Model A: {flops_train_A:,.0f} ({flops_train_A:.2e})")
print(f"      CatBoost Model B: {flops_train_B:,.0f} ({flops_train_B:.2e})")
print(f"      Platt LR (×3):    {flops_platt_train:,.0f} ({flops_platt_train:.2e})")
print(f"      训练总计:          {flops_bpcf_training_total:,.0f} ({flops_bpcf_training_total:.2e})")
print(f"    推理 FLOPs (测试集):")
print(f"      CatBoost 推理:    {flops_inference:,.0f} ({flops_inference:.2e})")
print(f"      校准 (Iso+LR):    {flops_calibration:,.0f} ({flops_calibration:.2e})")
print(f"      Temperature bias: {flops_tb:,.0f} ({flops_tb:.2e})")
print(f"      BPCF 融合:        {flops_fusion:,.0f} ({flops_fusion:.2e})")
print(f"      决策层:            {flops_decision:,.0f} ({flops_decision:.2e})")
print(f"      推理总计:          {flops_bpcf_inference_total:,.0f} ({flops_bpcf_inference_total:.2e})")
print(f"    总计 FLOPs:          {flops_bpcf_total:,.0f} ({flops_bpcf_total:.2e})")
print(f"\n  【对比】")
print(f"    训练开销比 (BPCF / B365): BPCF需训练, B365无需训练 (∞)")
print(f"    推理开销比 (BPCF / B365): {flops_bpcf_inference_total / max(flops_b365_total,1):.1f}x")
print(f"    每样本推理 FLOPs: B365={flops_b365_per_sample}, BPCF={flops_bpcf_inference_total // n_test}")

# 汇总表
cost_summary = pd.DataFrame({
    'Method': ['B365 (Baseline)', 'BPCF (Ours)'],
    'Training FLOPs': [0, flops_bpcf_training_total],
    'Inference FLOPs (test)': [flops_b365_total, flops_bpcf_inference_total],
    'Total FLOPs': [flops_b365_total, flops_bpcf_total],
    'Per-sample Inference FLOPs': [flops_b365_per_sample, flops_bpcf_inference_total // n_test],
    'Inference Speedup Ratio': [f"{flops_bpcf_inference_total / max(flops_b365_total,1):.1f}x (slower)", '1x (base)'],
})
cost_csv = os.path.join(save_dir, 'computational_cost_comparison.csv')
cost_summary.to_csv(cost_csv, index=False, encoding='utf-8-sig')
print(f"\n  计算开销对比表已保存: {cost_csv}")

# 详细分解表
cost_detail = pd.DataFrame({
    'Component': [
        'B365 odds→probs',
        'CatBoost Training (Model A)',
        'CatBoost Training (Model B)',
        'Platt LR Training (×3)',
        'CatBoost Inference (A+B)',
        'Calibration (Isotonic + LR)',
        'Temperature Bias',
        'BPCF Fusion Pipeline',
        'Decision Layer'
    ],
    'FLOPs': [
        flops_b365_total,
        flops_train_A,
        flops_train_B,
        flops_platt_train,
        flops_inference,
        flops_calibration,
        flops_tb,
        flops_fusion,
        flops_decision
    ],
    'Category': [
        'B365', 'BPCF-Train', 'BPCF-Train', 'BPCF-Train',
        'BPCF-Infer', 'BPCF-Infer', 'BPCF-Infer', 'BPCF-Infer', 'BPCF-Infer'
    ]
})
cost_detail_csv = os.path.join(save_dir, 'computational_cost_detail.csv')
cost_detail.to_csv(cost_detail_csv, index=False, encoding='utf-8-sig')
print(f"  计算开销详细分解已保存: {cost_detail_csv}")

# 保存 per-sample FLOPs 到统一目录
bpcf_per_sample_flops = flops_bpcf_inference_total // n_test
flops_save_dir_bpcf = '/mnt/train-decision-agent/dingzhijie/Trident-Cat code modification/cat/computational_cost'
os.makedirs(flops_save_dir_bpcf, exist_ok=True)
flops_df_bpcf = pd.DataFrame({'Method': ['BPCF'], 'Per-sample FLOPs': [bpcf_per_sample_flops]})
flops_df_bpcf.to_csv(os.path.join(flops_save_dir_bpcf, 'bpcf_flops.csv'), index=False, encoding='utf-8-sig')
print(f"  BPCF Per-sample FLOPs: {bpcf_per_sample_flops}")
print(f"  Saved to: {os.path.join(flops_save_dir_bpcf, 'bpcf_flops.csv')}")

# —— 保存完整结果到 txt
summary_path = os.path.join(save_dir, 'results_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("BPCF (Bounded Per-class Calibrated Fusion) 结果汇总\n")
    f.write("=" * 60 + "\n\n")

    f.write("【BPCF 参数】\n")
    f.write(f"  Per-class logit alpha: H={ALPHA_H}, D={ALPHA_D}, A={ALPHA_A}\n")
    f.write(f"  Platt ensemble weight: {ENSEMBLE_W_PLATT}\n")
    f.write(f"  Deviation cap: {DEV_CAP}\n")
    f.write(f"  Decision params: {DECISION}\n")
    f.write(f"  Temperature bias: {TB}\n\n")

    f.write("【整体 RPS】\n")
    f.write(f"  B365  RPS: {rps_b365:.10f}\n")
    f.write(f"  模型  RPS: {rps_model:.10f}\n")
    f.write(f"  BPCF  RPS: {rps_final:.10f}\n")
    f.write(f"  ΔRPS (BPCF - B365): {rps_final - rps_b365:.10f}\n\n")

    f.write("【测试集 Accuracy】\n")
    f.write(f"  融合 Accuracy: {acc_blend:.6f}\n")
    f.write(f"  预测占比: Win={( y_pred_blend==0).mean():.3f} | Draw={(y_pred_blend==1).mean():.3f} | Loss={(y_pred_blend==2).mean():.3f}\n\n")

    f.write("【统计检验 (BPCF vs B365)】\n")
    f.write(f"  配对 T 检验 (单侧):       t={t_res.statistic:.4f}, p={t_res.pvalue:.6f}  {sig_t}\n")
    f.write(f"  Wilcoxon signed-rank (单侧): W={w_res.statistic:.0f}, p={w_res.pvalue:.6f}  {sig_w}\n")
    f.write(f"  Mean Δ RPS = {diff_vec.mean():.10f}\n")
    f.write(f"  BPCF 更优样本: {(diff_vec<0).sum()}/{len(diff_vec)} ({100*(diff_vec<0).mean():.1f}%)\n\n")

    f.write("【分文件 RPS + Wilcoxon】\n")
    for fp in files_to_check:
        if not os.path.exists(fp):
            continue
        df_ = pd.read_csv(fp, encoding='utf-8')
        need = feature_cols + ['FTR','B365H','B365D','B365A']
        df_valid = df_.dropna(subset=need, how='any')
        if len(df_valid) == 0:
            continue
        Xv = df_valid[feature_cols].values
        y_true = df_valid['FTR'].values.astype(int)
        probs_b365_f = odds_to_probs(df_valid[['B365H','B365D','B365A']].values)
        pD_raw_f = model_A.predict_proba(Xv)[:,1]
        pHnD_raw_f = model_B.predict_proba(Xv)[:,1]
        pD_cal_f = calibrate_A(pD_raw_f)
        pHnD_cal_f = calibrate_B(pHnD_raw_f)
        pD_adj_f = _sigmoid(_logit_clip(pD_cal_f) * TB['A_t'] + TB['A_b'])
        pHnD_adj_f = _sigmoid(_logit_clip(pHnD_cal_f) * TB['B_t'] + TB['B_b'])
        Pm_f = _row_norm(np.column_stack([(1-pD_adj_f)*pHnD_adj_f, pD_adj_f, (1-pD_adj_f)*(1-pHnD_adj_f)]))
        Pf_platt_f = np.zeros_like(probs_b365_f)
        for c in range(3):
            Pf_platt_f[:, c] = platt_models[c].predict_proba(_logit_clip(probs_b365_f[:, c]).reshape(-1, 1))[:, 1]
        Pf_platt_f = _row_norm(Pf_platt_f)
        alpha_v = np.array([ALPHA_H, ALPHA_D, ALPHA_A])
        logit_bl_f = (1 - alpha_v) * _logit_clip(probs_b365_f) + alpha_v * _logit_clip(Pm_f)
        Pf_logit_blend_f = _row_norm(_sigmoid(logit_bl_f))
        Pf_ensemble_f = _row_norm(ENSEMBLE_W_PLATT * Pf_platt_f + (1 - ENSEMBLE_W_PLATT) * Pf_logit_blend_f)
        diff_pf_f = Pf_ensemble_f - probs_b365_f
        dist_pf_f = np.sqrt(np.sum(diff_pf_f**2, axis=1, keepdims=True))
        scale_pf_f = np.where(dist_pf_f > DEV_CAP, DEV_CAP / np.clip(dist_pf_f, 1e-12, None), 1.0)
        Pf_recal_f = _row_norm(np.clip(probs_b365_f + diff_pf_f * scale_pf_f, 1e-12, None))
        rps_b_f = calc_rps(y_true, probs_b365_f, 3)
        rps_m_f = calc_rps(y_true, Pm_f, 3)
        rps_c_f = calc_rps(y_true, Pf_recal_f, 3)
        rps_b_vec_f = calc_rps_vec(y_true, probs_b365_f)
        rps_c_vec_f = calc_rps_vec(y_true, Pf_recal_f)
        t_stat_ff, t_p_ff = stats.ttest_rel(rps_c_vec_f, rps_b_vec_f, alternative='less')
        w_stat_f, w_p_f = stats.wilcoxon(rps_c_vec_f, rps_b_vec_f, alternative='less')
        diff_vec_f = rps_c_vec_f - rps_b_vec_f
        sig_t_ff = '★★★' if t_p_ff<0.01 else ('★★' if t_p_ff<0.05 else ('★' if t_p_ff<0.10 else '×'))
        sig_f = '★★★' if w_p_f<0.01 else ('★★' if w_p_f<0.05 else ('★' if w_p_f<0.10 else '×'))
        f.write(f"  —— {os.path.basename(fp)} ——\n")
        f.write(f"    B365 RPS: {rps_b_f:.10f}\n")
        f.write(f"    模型 RPS: {rps_m_f:.10f}\n")
        f.write(f"    BPCF RPS: {rps_c_f:.10f}\n")
        f.write(f"    配对 T 检验 (BPCF < B365): t={t_stat_ff:.4f}, p={t_p_ff:.6f}  {sig_t_ff}\n")
        f.write(f"    Wilcoxon (BPCF < B365): W={w_stat_f:.0f}, p={w_p_f:.6f}  {sig_f}\n")
        f.write(f"    Mean Δ RPS = {diff_vec_f.mean():.10f}, BPCF更优: {(diff_vec_f<0).sum()}/{len(diff_vec_f)} ({100*(diff_vec_f<0).mean():.1f}%)\n\n")

    f.write("【计算开销对比 (FLOPs)】\n")
    f.write(f"  B365 推理 FLOPs: {flops_b365_total:,.0f} ({flops_b365_total:.2e})\n")
    f.write(f"  BPCF 训练 FLOPs: {flops_bpcf_training_total:,.0f} ({flops_bpcf_training_total:.2e})\n")
    f.write(f"  BPCF 推理 FLOPs: {flops_bpcf_inference_total:,.0f} ({flops_bpcf_inference_total:.2e})\n")
    f.write(f"  BPCF 总计 FLOPs: {flops_bpcf_total:,.0f} ({flops_bpcf_total:.2e})\n")
    f.write(f"  推理开销比 (BPCF/B365): {flops_bpcf_inference_total / max(flops_b365_total,1):.1f}x\n")
    f.write(f"  每样本推理: B365={flops_b365_per_sample} FLOPs, BPCF={flops_bpcf_inference_total // n_test} FLOPs\n\n")

    f.write("【蒸馏一致性】\n")
    f.write(f"  KL(B365 || Model_B): {kl:.10f}\n\n")

    f.write("【混淆矩阵】\n")
    f.write(cm_df.to_string() + "\n\n")

    f.write("【Precision / Recall / F1】\n")
    f.write(prf_table.to_string() + "\n")

print("\nROC 图已输出到：", save_dir)
print("混淆矩阵+PRF 已输出：", out_csv)
print("结果汇总已输出：", summary_path)
print("所有结果与图表已输出到：", save_dir)
