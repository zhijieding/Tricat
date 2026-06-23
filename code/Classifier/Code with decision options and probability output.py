"""
带决策方案和概率输出的代码 — BPCF (Bounded Per-class Calibrated Fusion)
导出: rps_per_match_test_with_argmax.csv（含决策打分列、概率对照列）
"""
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import shap
import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== 0) 复现性与路径 =====
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42); rng = np.random.default_rng(42)

folder_path   = '/mnt/train-decision-agent/dingzhijie/Trident-Cat code modification/数据/英超/赛季数据'
output_folder = '/mnt/train-decision-agent/dingzhijie/Trident-Cat code modification/数据/英超'
save_dir      = '/mnt/train-decision-agent/dingzhijie/Trident-Cat code modification/cat/英超'
os.makedirs(save_dir, exist_ok=True)

train_files = ['1415.csv','1516.csv','1617.csv','1718.csv','1819.csv','1920.csv','2021.csv','2122.csv']
test_files  = ['2223.csv','2324.csv']

# ===== 1) 固定参数 =====
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

# BPCF 参数
ALPHA_H = 0.42
ALPHA_D = 0.45
ALPHA_A = 0.93
ENSEMBLE_W_PLATT = 0.67
DEV_CAP = 0.065

DECISION = dict(alphaH=1.10, alphaD=1.30, alphaA=1.10, tauH=0.60, tauD=0.20, tauA=0.30, gamma=0.00)

# ===== 2) 数据读取与清洗 =====
columns_to_keep = [
    'Date','HomeTeam','AwayTeam','FTR',
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

def parse_date_safe(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, format='%d/%m/%Y', errors='coerce')
    mask = d.isna()
    if mask.any():
        d2 = pd.to_datetime(s[mask], format='%d/%m/%y', errors='coerce')
        d.loc[mask] = d2; mask = d.isna()
    if mask.any():
        d3 = pd.to_datetime(s[mask], format='%Y-%m-%d', errors='coerce')
        d.loc[mask] = d3; mask = d.isna()
    if mask.any():
        d4 = pd.to_datetime(s[mask], format='%Y/%m/%d', errors='coerce')
        d.loc[mask] = d4; mask = d.isna()
    if mask.any():
        d5 = pd.to_datetime(s[mask], format='%m/%d/%Y', errors='coerce')
        d.loc[mask] = d5; mask = d.isna()
    if mask.any():
        d6 = pd.to_datetime(s[mask], format='%m-%d-%Y', errors='coerce')
        d.loc[mask] = d6; mask = d.isna()
    if mask.any():
        d7 = pd.to_datetime(s[mask], errors='coerce', dayfirst=False)
        d.loc[mask] = d7
    return d

def _clean_and_save(src_folder, files, dst_folder):
    for file in sorted(files):
        fp = os.path.join(src_folder, file)
        if not os.path.exists(fp): continue
        df = pd.read_csv(fp, encoding='utf-8', na_values=NA, keep_default_na=True)
        df['FTR'] = df['FTR'].map({'H':0,'D':1,'A':2})
        df = df.dropna(subset=columns_to_keep, how='any')
        if 'Date' in df.columns:
            df['Date'] = parse_date_safe(df['Date'])
            df = df.dropna(subset=['Date']).sort_values('Date', kind='mergesort')
        out = os.path.join(dst_folder, f"df{file.split('.')[0]}.csv")
        df.to_csv(out, index=False, encoding='utf-8')

_clean_and_save(folder_path, train_files, train_output_folder)
_clean_and_save(folder_path, test_files,  test_output_folder)

train_dfs = [pd.read_csv(os.path.join(train_output_folder, f), encoding='utf-8')
             for f in sorted(os.listdir(train_output_folder)) if f.endswith('.csv')]
test_dfs  = [pd.read_csv(os.path.join(test_output_folder, f),  encoding='utf-8')
             for f in sorted(os.listdir(test_output_folder)) if f.endswith('.csv')]

train_data = pd.concat(train_dfs, ignore_index=True).fillna(0)
test_data  = pd.concat(test_dfs,  ignore_index=True).fillna(0)
train_data.to_csv(os.path.join(output_folder, 'train_data.csv'), index=False, encoding='utf-8')
test_data.to_csv(os.path.join(output_folder, 'test_data.csv'),   index=False, encoding='utf-8')

# ===== 3) 特征与工具函数 =====
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

def _row_norm(P): return P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)
def _logit_clip(p): p = np.clip(p, 1e-12, 1-1e-12); return np.log(p/(1-p))
def _sigmoid(x): return 1.0/(1.0+np.exp(-x))

def odds_to_probs(arr3):
    arr = arr3.astype(float)
    p = 1.0 / np.clip(arr, 1e-12, None)
    return _row_norm(p)

def calc_rps(y_true, y_prob, n_class=3):
    num = len(y_true); rps_sum = 0.0
    for i in range(num):
        pr = np.clip(y_prob[i], 1e-12, 1.0); pr = pr / pr.sum()
        oh = np.zeros(n_class); oh[int(y_true[i])] = 1.0
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

# ===== 4) 训练 A/B 头 + 校准 =====
split = int(0.9 * len(X_train_full))
X_fit, y_fit = X_train_full[:split], y_train_full[:split]
X_cal, y_cal = X_train_full[split:], y_train_full[split:]

train_b365 = odds_to_probs(train_df[['B365H','B365D','B365A']].values)
test_b365  = odds_to_probs(test_df [['B365H','B365D','B365A']].values)
b365_fit, b365_cal = train_b365[:split], train_b365[split:]

yA_fit = (y_fit == 1).astype(int)
softA_fit = b365_fit[:, 1]
XA_dup, yA_dup, wA_dup = build_soft_dataset(X_fit, yA_fit, softA_fit, lam=LAM_A)
model_A = CatBoostClassifier(**BEST_SHARED)
model_A.fit(XA_dup, yA_dup, sample_weight=wA_dup, verbose=False)

maskB_fit = (y_fit != 1)
XB_fit_raw = X_fit[maskB_fit]
yB_fit_hard = (y_fit[maskB_fit] == 0).astype(int)
teacher_fit_ha = b365_fit[maskB_fit][:, [0,2]]
teacher_fit_ha = teacher_fit_ha / teacher_fit_ha.sum(axis=1, keepdims=True)
softB_pos_fit  = teacher_fit_ha[:, 0]
XB_dup, yB_dup, wB_dup = build_soft_dataset(XB_fit_raw, yB_fit_hard, softB_pos_fit, lam=LAM_B)
model_B = CatBoostClassifier(**BEST_SHARED)
model_B.fit(XB_dup, yB_dup, sample_weight=wB_dup, verbose=False)

pD_cal_raw = model_A.predict_proba(X_cal)[:, 1]
yA_cal = (y_cal == 1).astype(int)
isoA = IsotonicRegression(out_of_bounds='clip').fit(pD_cal_raw, yA_cal)
def calibrate_A(p): return np.clip(isoA.predict(p), 1e-12, 1-1e-12)

maskB_cal = (y_cal != 1)
pHnD_cal_raw = model_B.predict_proba(X_cal[maskB_cal])[:, 1]
yB_cal = (y_cal[maskB_cal] == 0).astype(int)
if len(np.unique(yB_cal)) == 1:
    def calibrate_B(p): return p
else:
    z = _logit_clip(pHnD_cal_raw).reshape(-1,1)
    lrB = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000, class_weight='balanced').fit(z, yB_cal)
    def calibrate_B(p):
        zz = _logit_clip(p).reshape(-1,1)
        return lrB.predict_proba(zz)[:,1]

# ===== 5) 测试集模型概率 + BPCF融合 =====
pD_test_raw   = model_A.predict_proba(X_test_full)[:, 1]
pHnD_test_raw = model_B.predict_proba(X_test_full)[:, 1]
pD_test_cal   = calibrate_A(pD_test_raw)
pHnD_test_cal = calibrate_B(pHnD_test_raw)
pD_test_adj   = _sigmoid(_logit_clip(pD_test_cal)   * TB['A_t'] + TB['A_b'])
pHnD_test_adj = _sigmoid(_logit_clip(pHnD_test_cal) * TB['B_t'] + TB['B_b'])
prob_model_test = _row_norm(np.column_stack([
    (1 - pD_test_adj) * pHnD_test_adj,
    pD_test_adj,
    (1 - pD_test_adj) * (1 - pHnD_test_adj)
]))

# Platt scaling (fit on last 6 seasons)
n_skip = sum([len(d) for d in train_dfs[:2]])
platt_models = []
for c in range(3):
    z = _logit_clip(train_b365[n_skip:, c]).reshape(-1, 1)
    yc = (y_train_full[n_skip:] == c).astype(int)
    lr_c = LogisticRegression(C=1e4, solver='lbfgs', max_iter=1000).fit(z, yc)
    platt_models.append(lr_c)

# BPCF: Per-class Logit Blend + Platt Ensemble + Deviation Cap
P_platt_test = np.zeros_like(test_b365)
for c in range(3):
    P_platt_test[:, c] = platt_models[c].predict_proba(_logit_clip(test_b365[:, c]).reshape(-1, 1))[:, 1]
P_platt_test = _row_norm(P_platt_test)

alpha_vec = np.array([ALPHA_H, ALPHA_D, ALPHA_A])
logit_blend = (1 - alpha_vec) * _logit_clip(test_b365) + alpha_vec * _logit_clip(prob_model_test)
P_logit_blend_test = _row_norm(_sigmoid(logit_blend))

P_ensemble_test = _row_norm(ENSEMBLE_W_PLATT * P_platt_test + (1 - ENSEMBLE_W_PLATT) * P_logit_blend_test)

diff_p = P_ensemble_test - test_b365
dist = np.sqrt(np.sum(diff_p**2, axis=1, keepdims=True))
scale = np.where(dist > DEV_CAP, DEV_CAP / np.clip(dist, 1e-12, None), 1.0)
prob_blend_test = _row_norm(np.clip(test_b365 + diff_p * scale, 1e-12, None))

print(f"[BPCF] alpha_H={ALPHA_H}, alpha_D={ALPHA_D}, alpha_A={ALPHA_A}")
print(f"[BPCF] w_platt={ENSEMBLE_W_PLATT}, dev_cap={DEV_CAP}")

# ===== 6) 决策层 =====
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

prob_blend_test_dec = classwise_temp(prob_blend_test, DECISION['alphaH'], DECISION['alphaD'], DECISION['alphaA'])
y_pred_blend = predict_with_params(prob_blend_test_dec, DECISION['tauH'], DECISION['tauD'], DECISION['tauA'], DECISION['gamma'])

# ===== 7) 输出 =====
rps_b365   = calc_rps(y_test_full, test_b365, 3)
rps_model  = calc_rps(y_test_full, prob_model_test, 3)
rps_final  = calc_rps(y_test_full, prob_blend_test, 3)
acc_blend  = accuracy_score(y_test_full, y_pred_blend)

print(f"B365 test_data.csv 的RPS: {rps_b365}")
print(f"模型 test_data.csv 的RPS: {rps_model}")
print(f"BPCF 融合 test_data.csv 的RPS: {rps_final}")
print("[测试集预测占比] Win={:.3f} | Draw={:.3f} | Loss={:.3f}".format(
    (y_pred_blend==0).mean(), (y_pred_blend==1).mean(), (y_pred_blend==2).mean()
))
print("测试集 Accuracy：融合 =", acc_blend)

# ===== 8) 带决策方案的概率对照导出 =====
LABELS_WDL = np.array(["W", "D", "L"])

p_model = prob_model_test
p_blend = prob_blend_test
p_b365  = test_b365

pred_argmax_model = LABELS_WDL[p_model.argmax(axis=1)]
pred_argmax_blend = LABELS_WDL[p_blend.argmax(axis=1)]
pred_argmax_b365  = LABELS_WDL[p_b365.argmax(axis=1)]
pred_blend_wdl = LABELS_WDL[np.asarray(y_pred_blend).astype(int)]

base_cols = ["Date","HomeTeam","AwayTeam","FTR"]
df_out = test_df.loc[:, [c for c in base_cols if c in test_df.columns]].copy()
df_out["Pred_blend"]         = pred_blend_wdl
df_out["Pred_argmax_blend"]  = pred_argmax_blend
df_out["Pred_argmax_model"]  = pred_argmax_model
df_out["Pred_argmax_b365"]   = pred_argmax_b365

for i, lab in enumerate(LABELS_WDL):
    df_out[f"P_blend_{lab}"] = p_blend[:, i]
    df_out[f"P_model_{lab}"] = p_model[:, i]
    df_out[f"P_b365_{lab}"]  = p_b365[:,  i]

P_dec = prob_blend_test_dec
df_out["P_dec_W"] = P_dec[:, 0]
df_out["P_dec_D"] = P_dec[:, 1]
df_out["P_dec_L"] = P_dec[:, 2]

# 决策打分列
tauH = float(DECISION['tauH'])
tauD = float(DECISION['tauD'])
tauA = float(DECISION['tauA'])
gamma = float(DECISION.get('gamma', 0.0))
m_draw = 0.05

eps = 1e-12
df_out["S_H"] = P_dec[:, 0] / max(tauH, eps)
df_out["S_D"] = P_dec[:, 1] / max(tauD, eps)
df_out["S_A"] = P_dec[:, 2] / max(tauA, eps)

Ph, Pa = P_dec[:, 0], P_dec[:, 2]
denom = np.clip(Ph + Pa, eps, None)
df_out["p_home_cond"] = Ph / denom

df_out["tauH"]   = tauH
df_out["tauD"]   = tauD
df_out["tauA"]   = tauA
df_out["gamma"]  = gamma
df_out["m_draw"] = m_draw
df_out["alphaH"] = float(DECISION.get('alphaH', 1.0))
df_out["alphaD"] = float(DECISION.get('alphaD', 1.0))
df_out["alphaA"] = float(DECISION.get('alphaA', 1.0))

# 一致性检查
consistency = (np.asarray(y_pred_blend).astype(int) == p_blend.argmax(axis=1)).mean()
print(f"[Check] 强化决策 vs 融合 argmax 一致率：{consistency:.3f}")

csv_path = os.path.join(save_dir, "rps_per_match_test_with_argmax.csv")
df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
print("已导出对照CSV：", csv_path)

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

    pD_raw   = model_A.predict_proba(Xv)[:, 1]
    pHnD_raw = model_B.predict_proba(Xv)[:, 1]
    pD_cal   = calibrate_A(pD_raw)
    pHnD_cal = calibrate_B(pHnD_raw)
    pD_adj   = _sigmoid(_logit_clip(pD_cal)   * TB['A_t'] + TB['A_b'])
    pHnD_adj = _sigmoid(_logit_clip(pHnD_cal) * TB['B_t'] + TB['B_b'])
    Pm = _row_norm(np.column_stack([(1 - pD_adj) * pHnD_adj, pD_adj, (1 - pD_adj) * (1 - pHnD_adj)]))

    # BPCF融合
    Pf_platt = np.zeros_like(probs_b365)
    for c in range(3):
        Pf_platt[:, c] = platt_models[c].predict_proba(_logit_clip(probs_b365[:, c]).reshape(-1, 1))[:, 1]
    Pf_platt = _row_norm(Pf_platt)
    logit_bl = (1 - alpha_vec) * _logit_clip(probs_b365) + alpha_vec * _logit_clip(Pm)
    Pf_logit_blend = _row_norm(_sigmoid(logit_bl))
    Pf_ensemble = _row_norm(ENSEMBLE_W_PLATT * Pf_platt + (1 - ENSEMBLE_W_PLATT) * Pf_logit_blend)
    diff_pf = Pf_ensemble - probs_b365
    dist_pf = np.sqrt(np.sum(diff_pf**2, axis=1, keepdims=True))
    scale_pf = np.where(dist_pf > DEV_CAP, DEV_CAP / np.clip(dist_pf, 1e-12, None), 1.0)
    Pf_recal = _row_norm(np.clip(probs_b365 + diff_pf * scale_pf, 1e-12, None))

    print(f"\n—— {os.path.basename(file_path)} ——")
    print(f"B365  RPS: {calc_rps(y_true, probs_b365, 3)}")
    print(f"模型   RPS: {calc_rps(y_true, Pm, 3)}")
    print(f"BPCF  RPS: {calc_rps(y_true, Pf_recal, 3)}")

files_to_check = [
    os.path.join(test_output_folder, 'df2223.csv'),
    os.path.join(test_output_folder, 'df2324.csv'),
    os.path.join(output_folder,        'test_data.csv')
]
print("[分文件] 将评估：", [os.path.basename(x) for x in files_to_check])
for fp in files_to_check: per_file_rps(fp)

# —— 蒸馏一致性
maskB_test = y_test_full != 1
teacher_test_ha = test_b365[maskB_test][:, [0,2]]
teacher_test_ha = teacher_test_ha / teacher_test_ha.sum(axis=1, keepdims=True)
p_home_cond_kl = calibrate_B(model_B.predict_proba(X_test_full[maskB_test])[:,1])
pred_ha = np.vstack([p_home_cond_kl, 1 - p_home_cond_kl]).T
kl = np.mean(np.sum(teacher_test_ha * np.log((teacher_test_ha + eps)/(np.clip(pred_ha,eps,1) + eps)), axis=1))
print("测试集 蒸馏一致性 KL(B365 || Model_B)：", kl)

# —— 混淆矩阵 + PRF
cm = confusion_matrix(y_test_full, y_pred_blend, labels=[0,1,2])
labels3 = ['Home wins','Draws','Away wins']
cm_df = pd.DataFrame(cm, index=labels3, columns=['Predicted win','Predicted draw','Predicted loss'])
print("(a) confusion_matrix"); print(cm_df)

precision, recall, f1, _ = precision_recall_fscore_support(y_test_full, y_pred_blend, labels=[0,1,2], zero_division=0)
prf_table = (
    pd.DataFrame({'Precision': precision, 'Recall': recall, 'F1-score': f1}, index=labels3)
    .astype(float)
    .round(4)
)
print("(b) Precision-recall table"); print(prf_table)

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

# —— ROC + SHAP
def _plot_multiclass_roc(y_true, proba, title, save_path):
    y_bin = label_binarize(y_true, classes=[0,1,2])
    n_classes = y_bin.shape[1]
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), proba.ravel())
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
    plt.savefig(save_path, dpi=300); plt.close()

_plot_multiclass_roc(y_test_full, prob_model_test, 'ROC – Model (A&B calibrated + t,b)',
                     os.path.join(save_dir, 'roc_model_tb_both.png'))
_plot_multiclass_roc(y_test_full, prob_blend_test, 'ROC – BPCF (Bounded Per-class Calibrated Fusion)',
                     os.path.join(save_dir, 'roc_BPCF.png'))
print("ROC 图已输出到：", save_dir)

try:
    try:
        explainer_A = shap.TreeExplainer(model_A)
        shap_vals_A = explainer_A.shap_values(X_test_full)
        shap.summary_plot(shap_vals_A, X_test_full, feature_names=feature_cols,
                          plot_type='bar', max_display=30, show=False)
        plt.title('Task A (Draw vs Non-Draw) – SHAP bar')
        plt.savefig(os.path.join(save_dir, 'shap_taskA_bar.png'), dpi=300, bbox_inches='tight'); plt.close()

        shap.summary_plot(shap_vals_A, X_test_full, feature_names=feature_cols, show=False)
        plt.title('Task A (Draw vs Non-Draw) – SHAP beeswarm')
        plt.savefig(os.path.join(save_dir, 'shap_taskA_beeswarm.png'), dpi=300, bbox_inches='tight'); plt.close()
    except Exception as e:
        print('SHAP Task A 失败：', e)

    try:
        mask_non_draw = (y_test_full != 1)
        XB_te = X_test_full[mask_non_draw]
        explainer_B = shap.TreeExplainer(model_B)
        shap_vals_B = explainer_B.shap_values(XB_te)
        shap.summary_plot(shap_vals_B, XB_te, feature_names=feature_cols,
                          plot_type='bar', max_display=30, show=False)
        plt.title('Task B (Home vs Away | Non-Draw) – SHAP bar')
        plt.savefig(os.path.join(save_dir, 'shap_taskB_bar.png'), dpi=300, bbox_inches='tight'); plt.close()

        shap.summary_plot(shap_vals_B, XB_te, feature_names=feature_cols, show=False)
        plt.title('Task B (Home vs Away | Non-Draw) – SHAP beeswarm')
        plt.savefig(os.path.join(save_dir, 'shap_taskB_beeswarm.png'), dpi=300, bbox_inches='tight'); plt.close()
    except Exception as e:
        print('SHAP Task B 失败：', e)
except Exception as e:
    print("未能生成SHAP图（是否未安装 shap 包？）：", e)

print("混淆矩阵+PRF 已输出：", out_csv)
print("所有结果与图表已输出到：", save_dir)
