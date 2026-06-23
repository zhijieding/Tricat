from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_fscore_support
from catboost import CatBoostClassifier
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import shap
# =========================
# 一、数据处理
# =========================
folder_path = '/mnt/train-decision-agent/dingzhijie/Trident-Cat code modification/数据/赛季数据'
output_folder = '/mnt/train-decision-agent/dingzhijie/Trident-Cat code modification/数据'

train_files = [
    '1415.csv', '1516.csv', '1617.csv', '1718.csv', '1819.csv',
    '1920.csv', '2021.csv', '2122.csv',
]
test_files = ['2223.csv', '2324.csv']

columns_to_keep = [
    'Date', 'HomeTeam', 'AwayTeam', 'FTR',
    'Hform','Aform','Hst','ASt','HSTKPP','ASTKPP',
    'HGKPP','AGKPP','HCKPP','ACKPP','HAttack','AAttack',
    'HMidField','AMidField','HDefence','ADefense','HOverall','AOverall',
    'HTDG','ATDG','HStWeighted','AStWeighted','FormDifferential',
    'StDifferential','STKPP','GKPP','CKPP','RelAttack','RelMidField',
    'RelDefense','RelOverall','GDDifferential','StWeightedDifferential',
    'HS','AS','HF','AF','FDifferential','SDifferential',
    'B365H','B365D','B365A'
]

train_output_folder = os.path.join(output_folder, 'train')
test_output_folder  = os.path.join(output_folder, 'test')
os.makedirs(train_output_folder, exist_ok=True)
os.makedirs(test_output_folder,  exist_ok=True)

NA = ['', ' ', 'NA', 'N/A', 'na', 'NaN']

def parse_date_safe(s):
    d = pd.to_datetime(s, format='%d/%m/%Y', errors='coerce')
    for fmt in ['%d/%m/%y','%Y-%m-%d','%Y/%m/%d','%m/%d/%Y','%m-%d-%Y']:
        m = d.isna()
        if m.any():
            d.loc[m] = pd.to_datetime(s[m], format=fmt, errors='coerce')
    m = d.isna()
    if m.any():
        d.loc[m] = pd.to_datetime(s[m], errors='coerce', dayfirst=False)
    return d

def _clean_and_save(src_folder, files, dst_folder):
    for file in sorted(files):
        fp = os.path.join(src_folder, file)
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp, encoding='utf-8', na_values=NA, keep_default_na=True)
        df['FTR'] = df['FTR'].map({'H':0,'D':1,'A':2})
        df = df.dropna(subset=[c for c in columns_to_keep if c in df.columns], how='any')
        if 'Date' in df.columns:
            df['Date'] = parse_date_safe(df['Date'])
            df = df.dropna(subset=['Date']).sort_values('Date', kind='mergesort')
        df.to_csv(os.path.join(dst_folder, f"df{file.split('.')[0]}.csv"), index=False, encoding='utf-8')

_clean_and_save(folder_path, train_files, train_output_folder)
_clean_and_save(folder_path, test_files,  test_output_folder)

train_dfs = [pd.read_csv(os.path.join(train_output_folder, f), encoding='utf-8')
             for f in sorted(os.listdir(train_output_folder)) if f.endswith('.csv')]
train_data = pd.concat(train_dfs, ignore_index=True).fillna(0)
train_data.to_csv(os.path.join(output_folder, 'train_data.csv'), index=False, encoding='utf-8')

test_dfs = [pd.read_csv(os.path.join(test_output_folder, f), encoding='utf-8')
            for f in sorted(os.listdir(test_output_folder)) if f.endswith('.csv')]
test_data = pd.concat(test_dfs, ignore_index=True).fillna(0)
test_data.to_csv(os.path.join(output_folder, 'test_data.csv'), index=False, encoding='utf-8')

# =========================
# 二、准备数据与工具函数
# =========================
feature_cols = [
    'Hform','Aform','Hst','ASt','HSTKPP','ASTKPP',
    'HGKPP','AGKPP','HCKPP','ACKPP','HAttack','AAttack',
    'HMidField','AMidField','HDefence','ADefense','HOverall','AOverall',
    'HTDG','ATDG','HStWeighted','AStWeighted','FormDifferential',
    'StDifferential','STKPP','GKPP','CKPP','RelAttack','RelMidField',
    'RelDefense','RelOverall','GDDifferential','StWeightedDifferential',
    'HS','AS','HF','AF','FDifferential','SDifferential'
]

train_df = pd.read_csv(os.path.join(output_folder, 'train_data.csv'), encoding='utf-8')
test_df  = pd.read_csv(os.path.join(output_folder,  'test_data.csv'),  encoding='utf-8')

X_train_full = train_df[feature_cols].values
y_train_full = train_df['FTR'].values.astype(int)

X_test_full  = test_df[feature_cols].values
y_test_full  = test_df['FTR'].values.astype(int)

def odds_to_probs(arr3):
    arr = arr3.astype(float)
    p = 1.0 / np.clip(arr, 1e-12, None)
    p = p / p.sum(axis=1, keepdims=True)
    return p

train_b365 = odds_to_probs(train_df[['B365H','B365D','B365A']].values)
test_b365  = odds_to_probs(test_df [['B365H','B365D','B365A']].values)

def _row_norm(P):
    P = np.clip(P, 1e-12, None)
    return P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)

def calc_rps(y_true, y_prob, n_class=3):
    num = len(y_true)
    rps_sum = 0.0
    for i in range(num):
        pr = np.clip(y_prob[i], 1e-12, 1.0)
        pr = pr / pr.sum()
        oh = np.zeros(n_class, dtype=float)
        oh[int(y_true[i])] = 1.0
        cp = np.cumsum(pr); co = np.cumsum(oh)
        rps_sum += np.sum((cp - co) ** 2) / (n_class - 1)
    return rps_sum / max(num, 1)

def build_soft_dataset(X, y_hard, soft_pos, lam=0.7):
    y_smooth = (1 - lam) * y_hard + lam * soft_pos
    X_dup = np.repeat(X, 2, axis=0)
    y_dup = np.empty(2 * len(y_hard), dtype=int)
    w_dup = np.empty(2 * len(y_hard), dtype=float)
    y_dup[0::2] = 1; w_dup[0::2] = np.clip(y_smooth, 1e-6, 1-1e-6)
    y_dup[1::2] = 0; w_dup[1::2] = np.clip(1 - y_smooth, 1e-6, 1-1e-6)
    return X_dup, y_dup, w_dup

def _logit_clip(p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return np.log(p/(1-p))

def _sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def platt_fit(raw_p, y_binary, C=1e6):
    z = _logit_clip(raw_p).reshape(-1, 1)
    lr = LogisticRegression(C=C, solver='lbfgs', max_iter=1000, class_weight='balanced')
    lr.fit(z, y_binary.astype(int))
    return lr

def platt_apply(lr, raw_p):
    z = _logit_clip(raw_p).reshape(-1, 1)
    return lr.predict_proba(z)[:, 1]

# =========================
# 三、贝叶斯优化（双头蒸馏 + 共享超参）
# =========================
def cv_objective(iterations, depth, learning_rate, l2_leaf_reg,
                 bagging_temperature, random_strength, rsm, lam_A, lam_B):
    iterations = int(round(iterations))
    depth      = int(round(depth))
    shared_params = dict(
        iterations=iterations,
        depth=depth,
        learning_rate=float(learning_rate),
        l2_leaf_reg=float(l2_leaf_reg),
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        loss_function='Logloss',
        bootstrap_type='Bayesian',
        bagging_temperature=float(bagging_temperature),
        random_strength=float(random_strength),
        rsm=float(rsm),
    )

    lam_A = float(lam_A); lam_B = float(lam_B)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    rps_list = []

    for tr_idx, va_idx in skf.split(X_train_full, y_train_full):
        # A 头：Draw vs Non-Draw 蒸馏
        XA_tr = X_train_full[tr_idx]
        yA_tr_hard = (y_train_full[tr_idx] == 1).astype(int)
        softA_tr = train_b365[tr_idx][:, 1]
        XA_dup, yA_dup, wA_dup = build_soft_dataset(XA_tr, yA_tr_hard, softA_tr, lam=lam_A)
        model_A = CatBoostClassifier(**shared_params)
        model_A.fit(XA_dup, yA_dup, sample_weight=wA_dup, verbose=False)

        # B 头：Home vs Away 蒸馏（仅非平）
        maskB_tr = (y_train_full[tr_idx] != 1)
        XB_tr_raw = X_train_full[tr_idx][maskB_tr]
        yB_tr_hard = (y_train_full[tr_idx][maskB_tr] == 0).astype(int)
        teacher_tr_ha = train_b365[tr_idx][:, [0,2]][maskB_tr]
        teacher_tr_ha = teacher_tr_ha / teacher_tr_ha.sum(axis=1, keepdims=True)
        softB_pos = teacher_tr_ha[:, 0]  # P(H|非平)
        XB_dup, yB_dup, wB_dup = build_soft_dataset(XB_tr_raw, yB_tr_hard, softB_pos, lam=lam_B)
        model_B = CatBoostClassifier(**shared_params)
        model_B.fit(XB_dup, yB_dup, sample_weight=wB_dup, verbose=False)

        # 合成三类
        Xv = X_train_full[va_idx]
        yv = y_train_full[va_idx]
        pD_v   = model_A.predict_proba(Xv)[:, 1]
        pHnD_v = model_B.predict_proba(Xv)[:, 1]
        pH_v   = (1 - pD_v) * pHnD_v
        pA_v   = (1 - pD_v) * (1 - pHnD_v)
        probs3 = np.vstack([pH_v, pD_v, pA_v]).T

        rps = calc_rps(yv, probs3, 3)
        rps_list.append(rps)

    return -float(np.mean(rps_list))

pbounds = {
    'iterations': (400, 1500),
    'depth': (4, 10),
    'learning_rate': (0.01, 0.15),
    'l2_leaf_reg': (1.0, 20.0),
    'bagging_temperature': (0.0, 1.0),
    'random_strength': (0.5, 3.0),
    'rsm': (0.5, 1.0),
    'lam_A': (0.5, 0.95),
    'lam_B': (0.3, 0.85),
}

optimizer = BayesianOptimization(
    f=cv_objective,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)
optimizer.maximize(init_points=5, n_iter=20)
print("贝优-最优目标(= -CV_RPS):", optimizer.max['target'])
best_params_bo = optimizer.max['params']
print("贝优-最优参数(连续):", best_params_bo)

best_shared = dict(
    iterations=int(round(best_params_bo['iterations'])),
    depth=int(round(best_params_bo['depth'])),
    learning_rate=float(best_params_bo['learning_rate']),
    l2_leaf_reg=float(best_params_bo['l2_leaf_reg']),
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
    loss_function='Logloss',
    bootstrap_type='Bayesian',
    bagging_temperature=float(best_params_bo['bagging_temperature']),
    random_strength=float(best_params_bo['random_strength']),
    rsm=float(best_params_bo['rsm']),
    thread_count=1,
    task_type='CPU',
)
lam_A = float(best_params_bo['lam_A'])
lam_B = float(best_params_bo['lam_B'])
print("共享超参：", best_shared)
print(f"蒸馏混合：lam_A={lam_A:.3f}, lam_B={lam_B:.3f}")

# =========================
# 四、拆分训练/校准集，训练最终 A/B 模型
# =========================
split = int(0.9 * len(X_train_full))
X_fit, y_fit = X_train_full[:split], y_train_full[:split]
X_cal, y_cal = X_train_full[split:], y_train_full[split:]
b365_fit = train_b365[:split]
b365_cal = train_b365[split:]

# A 头（蒸馏）
yA_fit = (y_fit == 1).astype(int)
softA_fit = b365_fit[:, 1]
XA_dup, yA_dup, wA_dup = build_soft_dataset(X_fit, yA_fit, softA_fit, lam=lam_A)
model_A = CatBoostClassifier(**best_shared)
model_A.fit(XA_dup, yA_dup, sample_weight=wA_dup, verbose=False)

# B 头（蒸馏，仅非平）
maskB_fit = (y_fit != 1)
XB_fit_raw = X_fit[maskB_fit]
yB_fit_hard = (y_fit[maskB_fit] == 0).astype(int)
teacher_fit_ha = b365_fit[maskB_fit][:, [0,2]]
teacher_fit_ha = teacher_fit_ha / teacher_fit_ha.sum(axis=1, keepdims=True)
softB_pos_fit = teacher_fit_ha[:, 0]
XB_dup, yB_dup, wB_dup = build_soft_dataset(XB_fit_raw, yB_fit_hard, softB_pos_fit, lam=lam_B)
model_B = CatBoostClassifier(**best_shared)
model_B.fit(XB_dup, yB_dup, sample_weight=wB_dup, verbose=False)

# =========================
# 五、校准（A=Isotonic，B=Platt）
# =========================
# A：Isotonic
pD_cal_raw = model_A.predict_proba(X_cal)[:, 1]
yA_cal = (y_cal == 1).astype(int)
isoA = IsotonicRegression(out_of_bounds='clip')
isoA.fit(pD_cal_raw, yA_cal)
def calibrate_A(p): return np.clip(isoA.predict(p), 1e-12, 1-1e-12)

# B：Platt（非平拟合）
maskB_cal = (y_cal != 1)
pHnD_cal_raw = model_B.predict_proba(X_cal[maskB_cal])[:, 1]
yB_cal = (y_cal[maskB_cal] == 0).astype(int)

if len(np.unique(yB_cal)) == 1:
    def calibrate_B(p): return p
else:
    lrB = platt_fit(pHnD_cal_raw, yB_cal, C=1e6)
    def calibrate_B(p): return platt_apply(lrB, p)

# =========================
# 六、A&B (t,b) 修正（坐标下降，RPS最小）& 三类合成
# =========================
pD_cal = calibrate_A(pD_cal_raw)
pHnD_cal_all_raw = model_B.predict_proba(X_cal)[:, 1]
pHnD_cal = calibrate_B(pHnD_cal_all_raw)

t_grids = [
    np.linspace(0.6, 1.8, 25),
    np.linspace(0.85, 1.35, 21),
    np.linspace(0.95, 1.10, 16),
]
b_grids = [
    np.linspace(-2.0, 2.0, 41),
    np.linspace(-0.8, 0.8, 33),
    np.linspace(-0.3, 0.3, 25),
]
MAX_ITERS   = 7
IMPROVE_TOL = 5e-7

def _search_A(pD_base, pHnD_fixed, y_cal, t_grid, b_grid):
    best = (1.0, 0.0, 1e9, None)
    z = _logit_clip(pD_base)
    for t in t_grid:
        zt = z * t
        for b in b_grid:
            pD_adj = _sigmoid(zt + b)
            pH = (1 - pD_adj) * pHnD_fixed
            pA = (1 - pD_adj) * (1 - pHnD_fixed)
            probs = np.vstack([pH, pD_adj, pA]).T
            rps = calc_rps(y_cal, probs, 3)
            if rps < best[2]:
                best = (t, b, rps, pD_adj)
    return best

def _search_B(pHnD_base, pD_fixed, y_cal, t_grid, b_grid):
    best = (1.0, 0.0, 1e9, None)
    z = _logit_clip(pHnD_base)
    for t in t_grid:
        zt = z * t
        for b in b_grid:
            pHnD_adj = _sigmoid(zt + b)
            pH = (1 - pD_fixed) * pHnD_adj
            pA = (1 - pD_fixed) * (1 - pHnD_adj)
            probs = np.vstack([pH, pD_fixed, pA]).T
            rps = calc_rps(y_cal, probs, 3)
            if rps < best[2]:
                best = (t, b, rps, pHnD_adj)
    return best

best_tA, best_bA = 1.0, 0.0
best_tB, best_bB = 1.0, 0.0
pD_cur   = pD_cal.copy()
pHnD_cur = pHnD_cal.copy()

for level in range(len(t_grids)):
    t_grid, b_grid = t_grids[level], b_grids[level]
    last_rps = 1e9
    for it in range(MAX_ITERS):
        tA, bA, rpsA, pD_new = _search_A(pD_cal, pHnD_cur, y_cal, t_grid, b_grid)
        pD_cur = pD_new; best_tA, best_bA = tA, bA

        tB, bB, rpsB, pHnD_new = _search_B(pHnD_cal, pD_cur, y_cal, t_grid, b_grid)
        pHnD_cur = pHnD_new; best_tB, best_bB = tB, bB

        pH = (1 - pD_cur) * pHnD_cur
        pA = (1 - pD_cur) * (1 - pHnD_cur)
        probs = np.vstack([pH, pD_cur, pA]).T
        cur_rps = calc_rps(y_cal, probs, 3)
        print(f"[坐标下降][层{level+1} 迭代{it+1}] RPS={cur_rps:.6f} | A(t,b)=({best_tA:.3f},{best_bA:.3f}) B(t,b)=({best_tB:.3f},{best_bB:.3f})")
        if last_rps - cur_rps < IMPROVE_TOL:
            break
        last_rps = cur_rps

print(f"[校准集] 坐标下降收敛：A(t,b)=({best_tA:.3f},{best_bA:.3f}) | B(t,b)=({best_tB:.3f},{best_bB:.3f})")

# 校准集 - 最终模型概率（供融合学习）
pD_cal_adj   = pD_cur
pHnD_cal_adj = pHnD_cur
pH_cal_adj = (1 - pD_cal_adj) * pHnD_cal_adj
pA_cal_adj = (1 - pD_cal_adj) * (1 - pHnD_cal_adj)
prob_model_cal = np.vstack([pH_cal_adj, pD_cal_adj, pA_cal_adj]).T

# 测试集 - 应用同一组 (t,b)
pD_test_raw   = model_A.predict_proba(X_test_full)[:, 1]
pHnD_test_raw = model_B.predict_proba(X_test_full)[:, 1]
pD_test_cal   = calibrate_A(pD_test_raw)
pHnD_test_cal = calibrate_B(pHnD_test_raw)
pD_test_adj   = _sigmoid(_logit_clip(pD_test_cal)   * best_tA + best_bA)
pHnD_test_adj = _sigmoid(_logit_clip(pHnD_test_cal) * best_tB + best_bB)
pH_test_adj = (1 - pD_test_adj) * pHnD_test_adj
pA_test_adj = (1 - pD_test_adj) * (1 - pHnD_test_adj)
prob_model_test = _row_norm(np.column_stack([pH_test_adj, pD_test_adj, pA_test_adj]))

# =========================
# 七、BPCF 融合（Per-class Logit Blend + Platt Ensemble + Deviation Cap）
# =========================
ALPHA_H = 0.42
ALPHA_D = 0.45
ALPHA_A = 0.93
ENSEMBLE_W_PLATT = 0.67
DEV_CAP = 0.065

print(f"\n[BPCF] Per-class logit alpha: H={ALPHA_H}, D={ALPHA_D}, A={ALPHA_A}")
print(f"[BPCF] Platt ensemble weight: {ENSEMBLE_W_PLATT}")
print(f"[BPCF] Deviation cap: {DEV_CAP}")

# Platt scaling (fit on last 6 seasons of training data)
n_skip = sum([len(d) for d in train_dfs[:2]])
platt_models = []
for c in range(3):
    z = _logit_clip(train_b365[n_skip:, c]).reshape(-1, 1)
    yc = (y_train_full[n_skip:] == c).astype(int)
    lr_c = LogisticRegression(C=1e4, solver='lbfgs', max_iter=1000).fit(z, yc)
    platt_models.append(lr_c)
print("[BPCF] Platt models fitted on last 6 seasons")

def apply_bpcf(B, M):
    """BPCF: Per-class Logit Blend + Platt Ensemble + Deviation Cap"""
    # Part 1: B365 Platt scaling
    P_platt = np.zeros_like(B)
    for c in range(3):
        P_platt[:, c] = platt_models[c].predict_proba(_logit_clip(B[:, c]).reshape(-1, 1))[:, 1]
    P_platt = _row_norm(P_platt)

    # Part 2: Per-class logit-space blend (different α for H, D, A)
    alpha_vec = np.array([ALPHA_H, ALPHA_D, ALPHA_A])
    logit_blend = (1 - alpha_vec) * _logit_clip(B) + alpha_vec * _logit_clip(M)
    P_logit_blend = _row_norm(_sigmoid(logit_blend))

    # Part 3: Ensemble Platt + logit blend
    P_ensemble = _row_norm(ENSEMBLE_W_PLATT * P_platt + (1 - ENSEMBLE_W_PLATT) * P_logit_blend)

    # Part 4: Deviation cap
    diff_p = P_ensemble - B
    dist = np.sqrt(np.sum(diff_p**2, axis=1, keepdims=True))
    scale = np.where(dist > DEV_CAP, DEV_CAP / np.clip(dist, 1e-12, None), 1.0)
    Pf = _row_norm(np.clip(B + diff_p * scale, 1e-12, None))
    return Pf

# 校准集 BPCF 融合
prob_blend_cal = apply_bpcf(b365_cal, prob_model_cal)

# 测试集 BPCF 融合
prob_blend_test = apply_bpcf(test_b365, prob_model_test)

# 打印测试集 RPS
rps_b365  = calc_rps(y_test_full, test_b365)
rps_model = calc_rps(y_test_full, prob_model_test)
rps_final = calc_rps(y_test_full, prob_blend_test)
print(f"\nB365 test_data.csv 的RPS: {rps_b365}")
print(f"模型 test_data.csv 的RPS: {rps_model}")
print(f"BPCF 融合 test_data.csv 的RPS: {rps_final}")

# =========================
# 八、强化决策层：按类温度 α_H/α_D/α_A + 按类阈值 τ_H/τ_D/τ_A + H/A 间隔 γ
# =========================
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

def evaluate_decision(y_true, y_pred):
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0,1,2], zero_division=0
    )
    return float(f1.mean()), prec, rec, f1

def search_decision_params_strong(P_cal, y_cal):
    rate_H = float((y_cal == 0).mean())
    rate_D = float((y_cal == 1).mean())
    rate_A = float((y_cal == 2).mean())

    alpha_sets = [
        (1.0, 1.0, 1.0), (1.1, 1.3, 1.1), (1.2, 1.5, 1.2),
        (1.3, 1.6, 1.3), (1.0, 1.7, 1.2), (1.2, 1.2, 1.5)
    ]
    grid_H = np.linspace(0.30, 0.70, 9)
    grid_D = np.linspace(0.15, 0.60, 10)
    grid_A = np.linspace(0.30, 0.70, 9)
    gammas = np.linspace(0.00, 0.30, 16)

    floors = [
        (1.02, 1.10, 1.02),
        (1.00, 1.10, 1.00),
        (0.95, 1.05, 0.95),
        (0.90, 1.00, 0.90),
    ]
    prec_floor = 0.25

    best = None
    for (fH, fD, fA) in floors:
        best = (-1.0, (1.0,1.0,1.0), 0.5, 0.35, 0.5, 0.10)
        for (aH, aD, aA) in alpha_sets:
            P_dec = classwise_temp(P_cal, aH, aD, aA)
            for th in grid_H:
                for td in grid_D:
                    for ta in grid_A:
                        for g in gammas:
                            y_pred = predict_with_params(P_dec, th, td, ta, g)
                            ph = float((y_pred == 0).mean())
                            pd = float((y_pred == 1).mean())
                            pa = float((y_pred == 2).mean())

                            if (ph < fH * rate_H) or (pd < fD * rate_D) or (pa < fA * rate_A):
                                continue
                            _, prec, _, _ = precision_recall_fscore_support(
                                y_cal, y_pred, labels=[0,1,2], zero_division=0
                            )
                            if (prec[0] < prec_floor) or (prec[1] < prec_floor) or (prec[2] < prec_floor):
                                continue
                            score, _, _, _ = evaluate_decision(y_cal, y_pred)
                            if score > best[0]:
                                best = (score, (aH,aD,aA), float(th), float(td), float(ta), float(g))
        if best[0] >= 0:
            break

    if best[0] < 0:
        for (aH, aD, aA) in alpha_sets:
            P_dec = classwise_temp(P_cal, aH, aD, aA)
            for th in grid_H:
                for td in grid_D:
                    for ta in grid_A:
                        for g in gammas:
                            y_pred = predict_with_params(P_dec, th, td, ta, g)
                            score, _, _, _ = evaluate_decision(y_cal, y_pred)
                            if (best is None) or (score > best[0]):
                                best = (score, (aH,aD,aA), float(th), float(td), float(ta), float(g))

    score, (aH,aD,aA), tau_H, tau_D, tau_A, gamma = best
    print(f"[决策参数] α_H={aH:.2f}, α_D={aD:.2f}, α_A={aA:.2f} | τ_H={tau_H:.2f}, τ_D={tau_D:.2f}, τ_A={tau_A:.2f} | γ={gamma:.2f} | 校准集宏F1={score:.4f}")
    return (aH,aD,aA), tau_H, tau_D, tau_A, gamma

# —— 在校准集上学习决策参数
(alH, alD, alA), tau_H, tau_D, tau_A, gamma = search_decision_params_strong(prob_blend_cal, y_cal)

# —— 在测试集上应用
prob_blend_test_dec = classwise_temp(prob_blend_test, alH, alD, alA)
y_pred_blend = predict_with_params(prob_blend_test_dec, tau_H, tau_D, tau_A, gamma)

print("[测试集预测占比] Win={:.3f} | Draw={:.3f} | Loss={:.3f}".format(
    (y_pred_blend==0).mean(), (y_pred_blend==1).mean(), (y_pred_blend==2).mean()
))

# =========================
# 九、整体指标与分文件 RPS
# =========================
acc_blend = accuracy_score(y_test_full, y_pred_blend)
print("测试集 Accuracy：融合 =", acc_blend)

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

    # B365 概率
    probs_b365 = odds_to_probs(df_valid[['B365H','B365D','B365A']].values)

    # 模型三类概率（双头 + 校准 + t,b）
    pD_raw   = model_A.predict_proba(Xv)[:, 1]
    pHnD_raw = model_B.predict_proba(Xv)[:, 1]
    pD_cal_  = calibrate_A(pD_raw)
    pHnD_cal_= calibrate_B(pHnD_raw)
    pD_adj   = _sigmoid(_logit_clip(pD_cal_)   * best_tA + best_bA)
    pHnD_adj = _sigmoid(_logit_clip(pHnD_cal_) * best_tB + best_bB)
    Pm = _row_norm(np.column_stack([
        (1 - pD_adj) * pHnD_adj,
        pD_adj,
        (1 - pD_adj) * (1 - pHnD_adj)
    ]))

    # BPCF 融合
    Pf_recal = apply_bpcf(probs_b365, Pm)

    print(f"\n—— {os.path.basename(file_path)} ——")
    print(f"B365  RPS: {calc_rps(y_true, probs_b365, 3)}")
    print(f"模型   RPS: {calc_rps(y_true, Pm, 3)}")
    print(f"BPCF  RPS: {calc_rps(y_true, Pf_recal, 3)}")

# —— 明确列出三份"测试集"文件并调用
files_to_check = [
    os.path.join(test_output_folder, 'df2223.csv'),
    os.path.join(test_output_folder, 'df2324.csv'),
    os.path.join(output_folder,        'test_data.csv')
]
print("[分文件] 将评估：", [os.path.basename(x) for x in files_to_check])
for fp in files_to_check:
    per_file_rps(fp)

# =========================
# 十、蒸馏一致性（测试集非平样本 KL）
# =========================
maskB_test = y_test_full != 1
teacher_test_ha = test_b365[maskB_test][:, [0,2]]
teacher_test_ha = teacher_test_ha / teacher_test_ha.sum(axis=1, keepdims=True)
p_home_cond = calibrate_B(model_B.predict_proba(X_test_full[maskB_test])[:,1])
pred_ha = np.vstack([p_home_cond, 1 - p_home_cond]).T
eps = 1e-12
kl = np.mean(np.sum(teacher_test_ha * np.log((teacher_test_ha + eps)/(np.clip(pred_ha,eps,1) + eps)), axis=1))
print("\n测试集 蒸馏一致性 KL(B365 || Model_B)：", kl)

# =========================
# 十一、混淆矩阵 + PRF（基于强化决策层的最终类别）
# =========================
cm = confusion_matrix(y_test_full, y_pred_blend, labels=[0,1,2])
labels3 = ['Home wins','Draws','Away wins']
cm_df = pd.DataFrame(cm, index=labels3, columns=['Predicted win','Predicted draw','Predicted loss'])
print("\n(a) confusion_matrix"); print(cm_df)

precision, recall, f1, _ = precision_recall_fscore_support(y_test_full, y_pred_blend, labels=[0,1,2], zero_division=0)
prf_table = pd.DataFrame({'Precision':precision,'Recall':recall,'F1-score':f1}, index=labels3).applymap(lambda x: round(float(x),4))
print("(b) Precision-recall table"); print(prf_table)

save_dir = '/mnt/train-decision-agent/dingzhijie/Trident-Cat code modification/cat'
os.makedirs(save_dir, exist_ok=True)
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
print("混淆矩阵+PRF 已输出：", out_csv)

# =========================
# 十二、ROC（模型 / 融合）
# =========================
def plot_multiclass_roc(y_true, y_score, title, out_path):
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = y_bin.shape[1]
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
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
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(out_path, dpi=300); plt.close()

plot_multiclass_roc(y_test_full, prob_model_test, 'ROC – Model (calibrated + A&B t,b)',
                    os.path.join(save_dir, 'roc_model_tb_both.png'))
plot_multiclass_roc(y_test_full, prob_blend_test, 'ROC – BPCF (Bounded Per-class Calibrated Fusion)',
                    os.path.join(save_dir, 'roc_BPCF.png'))
print("ROC 图已输出到：", save_dir)

# =========================
# 十三、SHAP（A / B 头）
# =========================
try:
    try:
        explainer_A = shap.TreeExplainer(model_A)
        shap_vals_A = explainer_A.shap_values(X_test_full)
        shap.summary_plot(shap_vals_A, X_test_full, feature_names=feature_cols, plot_type='bar', max_display=30, show=False)
        plt.title('Task A (Draw vs Non-Draw) – SHAP bar')
        plt.savefig(os.path.join(save_dir, 'shap_taskA_bar.png'), dpi=300, bbox_inches='tight'); plt.close()
        shap.summary_plot(shap_vals_A, X_test_full, feature_names=feature_cols, show=False)
        plt.title('Task A (Draw vs Non-Draw) – SHAP beeswarm')
        plt.savefig(os.path.join(save_dir, 'shap_taskA_beeswarm.png'), dpi=300, bbox_inches='tight'); plt.close()
    except Exception as e:
        print('SHAP Task A 失败：', e)

    try:
        maskB_test_only = (y_test_full != 1)
        XB_test = X_test_full[maskB_test_only]
        explainer_B = shap.TreeExplainer(model_B)
        shap_vals_B = explainer_B.shap_values(XB_test)
        shap.summary_plot(shap_vals_B, XB_test, feature_names=feature_cols, plot_type='bar', max_display=30, show=False)
        plt.title('Task B (Home vs Away | Non-Draw) – SHAP bar')
        plt.savefig(os.path.join(save_dir, 'shap_taskB_bar.png'), dpi=300, bbox_inches='tight'); plt.close()
        shap.summary_plot(shap_vals_B, XB_test, feature_names=feature_cols, show=False)
        plt.title('Task B (Home vs Away | Non-Draw) – SHAP beeswarm')
        plt.savefig(os.path.join(save_dir, 'shap_taskB_beeswarm.png'), dpi=300, bbox_inches='tight'); plt.close()
    except Exception as e:
        print('SHAP Task B 失败：', e)
except Exception as e:
    print("未能生成SHAP图（是否未安装 shap 包？）：", e)

print("\n所有结果与图表已输出到：", save_dir)
