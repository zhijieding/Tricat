# -*- coding: utf-8 -*-
"""
对比模型: BalanceNet (改编自 Ghost Recon Online 防恐精英模型)

原始论文方法:
- 输入: 每个玩家的统计特征 + 可学习Embedding
- 玩家表示层: W × 统计特征 → 低维表示, 加上Embedding得到玩家向量
- 队伍表示: 同队玩家向量求和 → 队伍向量
- 隐藏层: V_A t_A + V_B t_B + b → Tanh
- 输出层: Sigmoid → P(A队获胜)

本代码改编为足球比赛预测:
- 将主队特征视为"A队"，客队特征视为"B队"
- 主队统计特征 → 线性投影 → 主队向量 V_H
- 客队统计特征 → 线性投影 → 客队向量 V_A
- 拼接 [V_H; V_A] → 隐藏层(Tanh) → 输出层(Softmax, 3分类: H/D/A)
- 使用贝叶斯优化调参
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.pipeline import Pipeline
from bayes_opt import BayesianOptimization
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ===== 配置 =====
LEAGUE = os.environ.get('BALANCENET_LEAGUE', '英超')
BASE_DIR = '/mnt/train-decision-agent/dingzhijie/Trident-Cat code modification'
DATA_DIR = os.path.join(BASE_DIR, '数据', '7.final data', LEAGUE)
SAVE_DIR = os.path.join(BASE_DIR, 'cat', LEAGUE, 'run_BalanceNet')
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"{'='*60}")
print(f"BalanceNet 对比模型 — 联赛: {LEAGUE}")
print(f"{'='*60}")

# ===== 特征定义 (与 7.final data 数据集对齐, 39个特征) =====
# 主队特征 (13个)
home_features = [
    'HDPI', 'HSC', 'HSOT', 'HG', 'HCK',
    'HATT', 'HMID', 'HOVR', 'HGDT', 'HWS',
    'HS', 'HF', 'HDEF',
]
# 客队特征 (13个)
away_features = [
    'ADPI', 'ASC', 'ASOT', 'AG', 'ACK',
    'AATT', 'AMID', 'AOVR', 'AGDT', 'AWS',
    'AS', 'AF', 'ADEF',
]
# 差异特征 (13个)
diff_features = [
    'DPIDiff', 'STRDiff', 'STDiff', 'GDiff', 'CKDiff',
    'ARDiff', 'MRDiff', 'DRDiff', 'ORDiff', 'GDDiff',
    'WSDiff', 'FDiff', 'SDiff',
]
# 赔率特征
odds_features = ['B365H', 'B365D', 'B365A']

feature_cols = home_features + away_features + diff_features
all_needed_cols = feature_cols + odds_features + ['FTR', 'Date', 'HomeTeam', 'AwayTeam']

# ===== 数据加载 =====
train_files = ['1415.csv', '1516.csv', '1617.csv', '1718.csv',
               '1819.csv', '1920.csv', '2021.csv', '2122.csv']
test_files = ['2223.csv', '2324.csv']

def load_and_clean(files):
    dfs = []
    for f in files:
        fp = os.path.join(DATA_DIR, f)
        if not os.path.exists(fp):
            print(f"  [WARN] 文件不存在: {fp}")
            continue
        df = pd.read_csv(fp, encoding='utf-8')
        df['FTR'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
        available = [c for c in feature_cols + odds_features + ['FTR'] if c in df.columns]
        df = df.dropna(subset=available, how='any')
        dfs.append(df)
    return dfs

print("\n[数据加载]")
train_dfs = load_and_clean(train_files)
test_dfs = load_and_clean(test_files)
train_data = pd.concat(train_dfs, ignore_index=True).fillna(0)
test_data = pd.concat(test_dfs, ignore_index=True).fillna(0)
print(f"  训练集: {len(train_data)} 场比赛 ({len(train_dfs)} 个赛季)")
print(f"  测试集: {len(test_data)} 场比赛 ({len(test_dfs)} 个赛季)")

X_train_raw = train_data[feature_cols].values
y_train_raw = train_data['FTR'].values.astype(int)
X_test = test_data[feature_cols].values
y_test = test_data['FTR'].values.astype(int)

# 随机上采样处理类别不平衡
ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train_raw, y_train_raw)
print(f"  上采样后训练集: {len(X_train)} 样本")

# ===== RPS 计算函数 =====
def calc_rps(y_true, y_prob, n_class=3):
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob = np.clip(y_prob, 1e-12, None)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    n = len(y_true)
    rps_sum = 0.0
    for i in range(n):
        oh = np.zeros(n_class)
        oh[int(y_true[i])] = 1.0
        cp = np.cumsum(y_prob[i])
        co = np.cumsum(oh)
        rps_sum += np.sum((cp - co) ** 2) / (n_class - 1)
    return rps_sum / max(n, 1)

def calc_rps_vec(y_true, y_prob):
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob = np.clip(y_prob, 1e-12, None)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    n = len(y_true)
    rps = np.zeros(n)
    for i in range(n):
        oh = np.zeros(3)
        oh[int(y_true[i])] = 1.0
        rps[i] = np.sum((np.cumsum(y_prob[i]) - np.cumsum(oh)) ** 2) / 2.0
    return rps

def odds_to_probs(arr3):
    p = 1.0 / np.clip(arr3.astype(float), 1e-12, None)
    return p / p.sum(axis=1, keepdims=True)

# ===== RPS Scorer for CV =====
from sklearn.metrics import make_scorer

def rps_scorer(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    return calc_rps(y_true, y_prob)

neg_rps_scorer = make_scorer(rps_scorer, response_method="predict_proba",
                             greater_is_better=False)

# ===== 贝叶斯优化: BalanceNet (MLP with specific architecture) =====
import json

params_file = os.path.join(SAVE_DIR, 'best_params_BalanceNet.json')

if os.path.exists(params_file):
    print("\n[加载已有参数] 跳过贝叶斯优化...")
    with open(params_file, 'r', encoding='utf-8') as pf:
        saved = json.load(pf)
    best_params = {
        'hidden_layer_sizes': tuple(saved['hidden_layer_sizes']),
        'alpha': saved['alpha'],
        'learning_rate_init': saved['learning_rate_init'],
        'batch_size': saved['batch_size'],
        'solver': 'adam',
        'activation': 'tanh',
        'max_iter': 2000,
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.15,
        'n_iter_no_change': 10,
        'tol': 0.001
    }
else:
    print("\n[贝叶斯优化] 搜索 BalanceNet 最优超参数...")

    def balancenet_evaluate(layer1_size, layer2_size, alpha, learning_rate_init, batch_size):
        params = {
            'hidden_layer_sizes': (int(layer1_size), int(layer2_size)),
            'alpha': alpha,
            'learning_rate_init': learning_rate_init,
            'batch_size': int(batch_size),
            'solver': 'adam',
            'activation': 'tanh',
            'max_iter': 2000,
            'random_state': 42,
            'early_stopping': True,
            'validation_fraction': 0.15,
            'n_iter_no_change': 10,
            'tol': 0.001
        }

        model = MLPClassifier(**params)
        cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])
        cv_score = cross_val_score(pipe, X_train, y_train,
                                   scoring=neg_rps_scorer, cv=cv, n_jobs=-1).mean()
        return cv_score

    pbounds = {
        'layer1_size': (8, 64),
        'layer2_size': (16, 128),
        'alpha': (0.01, 10.0),
        'learning_rate_init': (1e-4, 5e-3),
        'batch_size': (32, 256)
    }

    optimizer = BayesianOptimization(
        f=balancenet_evaluate,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )

    optimizer.maximize(init_points=10, n_iter=30)

    best_params_raw = optimizer.max['params']
    best_params = {
        'hidden_layer_sizes': (int(best_params_raw['layer1_size']), int(best_params_raw['layer2_size'])),
        'alpha': best_params_raw['alpha'],
        'learning_rate_init': best_params_raw['learning_rate_init'],
        'batch_size': int(best_params_raw['batch_size']),
        'solver': 'adam',
        'activation': 'tanh',
        'max_iter': 2000,
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.15,
        'n_iter_no_change': 10,
        'tol': 0.001
    }

    with open(params_file, 'w', encoding='utf-8') as pf:
        json.dump({
            'hidden_layer_sizes': list(best_params['hidden_layer_sizes']),
            'alpha': best_params['alpha'],
            'learning_rate_init': best_params['learning_rate_init'],
            'batch_size': best_params['batch_size'],
        }, pf, indent=2)
    print(f"  参数已保存到: {params_file}")

print(f"\n[最优参数]")
print(f"  隐藏层结构: {best_params['hidden_layer_sizes']}")
print(f"  Alpha (L2): {best_params['alpha']:.6f}")
print(f"  学习率: {best_params['learning_rate_init']:.6f}")
print(f"  Batch size: {best_params['batch_size']}")
if 'optimizer' in dir():
    print(f"  最优CV RPS: {abs(optimizer.max['target']):.6f}")

# ===== 训练最终模型 =====
print("\n[训练最终模型]")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

best_model = MLPClassifier(**best_params)
best_model.fit(X_train_scaled, y_train)

# ===== 测试集评估 =====
print(f"\n{'='*60}")
print("测试集评估结果")
print(f"{'='*60}")

y_prob = best_model.predict_proba(X_test_scaled)
y_pred = best_model.predict(X_test_scaled)

# B365基准
b365_probs = odds_to_probs(test_data[odds_features].values)

# RPS
rps_model = calc_rps(y_test, y_prob)
rps_b365 = calc_rps(y_test, b365_probs)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n  B365 RPS:       {rps_b365:.10f}")
print(f"  BalanceNet RPS: {rps_model:.10f}")
print(f"  ΔRPS (模型-B365): {rps_model - rps_b365:.10f}")
print(f"  BalanceNet Accuracy: {accuracy:.6f}")
print(f"  预测占比: Win={( y_pred==0).mean():.3f} | Draw={(y_pred==1).mean():.3f} | Loss={(y_pred==2).mean():.3f}")

# ===== 统计检验 =====
print(f"\n{'='*60}")
print("统计检验 (BalanceNet vs B365)")
print(f"{'='*60}")

rps_model_vec = calc_rps_vec(y_test, y_prob)
rps_b365_vec = calc_rps_vec(y_test, b365_probs)
diff_vec = rps_model_vec - rps_b365_vec

t_res = stats.ttest_rel(rps_model_vec, rps_b365_vec, alternative='less')
w_res = stats.wilcoxon(rps_model_vec, rps_b365_vec, alternative='less')

sig_t = '★★★(p<1%)' if t_res.pvalue < 0.01 else ('★★(p<5%)' if t_res.pvalue < 0.05 else ('★(p<10%)' if t_res.pvalue < 0.10 else '×(不显著)'))
sig_w = '★★★(p<1%)' if w_res.pvalue < 0.01 else ('★★(p<5%)' if w_res.pvalue < 0.05 else ('★(p<10%)' if w_res.pvalue < 0.10 else '×(不显著)'))

print(f"  配对 T 检验 (单侧): t={t_res.statistic:.4f}, p={t_res.pvalue:.6f}  {sig_t}")
print(f"  Wilcoxon signed-rank (单侧): W={w_res.statistic:.0f}, p={w_res.pvalue:.6f}  {sig_w}")
print(f"  Mean Δ RPS = {diff_vec.mean():.10f}")
print(f"  BalanceNet 更优样本: {(diff_vec<0).sum()}/{len(diff_vec)} ({100*(diff_vec<0).mean():.1f}%)")

# ===== 每场 RPS 保存 =====
print(f"\n{'='*60}")
print("保存每场 RPS")
print(f"{'='*60}")

rps_per_match_df = pd.DataFrame({
    'Date': test_data['Date'].values,
    'HomeTeam': test_data['HomeTeam'].values,
    'AwayTeam': test_data['AwayTeam'].values,
    'FTR_true': y_test,
    'Pred_BalanceNet': y_pred,
    'RPS_B365': rps_b365_vec,
    'RPS_BalanceNet': rps_model_vec,
    'P_H_b365': b365_probs[:, 0],
    'P_D_b365': b365_probs[:, 1],
    'P_A_b365': b365_probs[:, 2],
    'P_H_BalanceNet': y_prob[:, 0],
    'P_D_BalanceNet': y_prob[:, 1],
    'P_A_BalanceNet': y_prob[:, 2],
})
rps_per_match_path = os.path.join(SAVE_DIR, 'rps_per_match_test.csv')
rps_per_match_df.to_csv(rps_per_match_path, index=False, encoding='utf-8-sig')
print(f"  已保存: {rps_per_match_path}")
print(f"  共 {len(rps_per_match_df)} 场比赛")

# ===== 分赛季 RPS =====
print(f"\n{'='*60}")
print("分赛季 RPS")
print(f"{'='*60}")

season_start = 0
for i, df in enumerate(test_dfs):
    n = len(df)
    season_end = season_start + n
    y_s = y_test[season_start:season_end]
    p_s = y_prob[season_start:season_end]
    b_s = b365_probs[season_start:season_end]

    rps_m = calc_rps(y_s, p_s)
    rps_b = calc_rps(y_s, b_s)

    rps_m_vec = calc_rps_vec(y_s, p_s)
    rps_b_vec = calc_rps_vec(y_s, b_s)
    d = rps_m_vec - rps_b_vec
    t_s, t_p = stats.ttest_rel(rps_m_vec, rps_b_vec, alternative='less')
    w_s, w_p = stats.wilcoxon(rps_m_vec, rps_b_vec, alternative='less')
    sig = '★★★' if w_p < 0.01 else ('★★' if w_p < 0.05 else ('★' if w_p < 0.10 else '×'))

    print(f"\n  —— {test_files[i]} ({n} 场) ——")
    print(f"    B365 RPS:       {rps_b:.10f}")
    print(f"    BalanceNet RPS: {rps_m:.10f}")
    print(f"    T检验: t={t_s:.4f}, p={t_p:.6f}")
    print(f"    Wilcoxon: p={w_p:.6f} {sig}")
    print(f"    Mean Δ = {d.mean():.10f}, 模型更优: {(d<0).sum()}/{n} ({100*(d<0).mean():.1f}%)")

    season_start = season_end

# ===== 混淆矩阵 + PRF =====
print(f"\n{'='*60}")
print("混淆矩阵 & Precision/Recall/F1")
print(f"{'='*60}")

cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
labels3 = ['Home wins', 'Draws', 'Away wins']
cm_df = pd.DataFrame(cm, index=labels3, columns=['Predicted win', 'Predicted draw', 'Predicted loss'])
print("\n(a) Confusion Matrix")
print(cm_df)

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1, 2], zero_division=0)
prf_table = pd.DataFrame({'Precision': precision, 'Recall': recall, 'F1-score': f1},
                         index=labels3).map(lambda x: round(float(x), 4))
print("\n(b) Precision-Recall Table")
print(prf_table)

# 保存结果表
out_csv = os.path.join(SAVE_DIR, 'result_table.csv')
sep1 = pd.DataFrame([[""] * 3], columns=cm_df.columns, index=[""])
sep2 = pd.DataFrame([[""] * 3], columns=prf_table.columns, index=[""])
upper = pd.concat([cm_df, sep1], axis=0)
lower = pd.concat([sep2, prf_table], axis=0)
result_table = pd.concat([upper, lower], axis=1)
result_table.columns = pd.MultiIndex.from_tuples(
    [("(a) Confusion matrix", c) if c in cm_df.columns else ("", c) for c in result_table.columns[:3]] +
    [("(b) Precision-recall table", c) if c in prf_table.columns else ("", c) for c in result_table.columns[3:]]
)
result_table.to_csv(out_csv, encoding='utf-8-sig')

# ===== ROC 曲线 =====
def plot_multiclass_roc(y_true, proba, title, save_path):
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
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

    plt.figure(figsize=(8, 6))
    plt.plot(fpr["micro"], tpr["micro"], label=f'micro-average (AUC={roc_auc["micro"]:.3f})')
    plt.plot(fpr["macro"], tpr["macro"], linestyle='--', label=f'macro-average (AUC={roc_auc["macro"]:.3f})')
    for i, name in enumerate(['Home', 'Draw', 'Away']):
        plt.plot(fpr[i], tpr[i], label=f'{name} (AUC={roc_auc[i]:.3f})')
    plt.plot([0, 1], [0, 1], linestyle=':')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(save_path, dpi=300)
    plt.close()

plot_multiclass_roc(y_test, y_prob, f'ROC – BalanceNet ({LEAGUE})',
                    os.path.join(SAVE_DIR, 'roc_BalanceNet.png'))

# ===== 计算开销对比 (FLOPs) =====
print(f"\n{'='*60}")
print("计算开销对比 (Computational Cost / FLOPs)")
print(f"{'='*60}")

n_features = len(feature_cols)
n_train = len(X_train)
n_test_samples = len(X_test)
l1 = best_params['hidden_layer_sizes'][0]
l2 = best_params['hidden_layer_sizes'][1]
n_output = 3

# B365: 10 FLOPs/sample
flops_b365 = 10 * n_test_samples

# BalanceNet Training FLOPs (MLP backprop)
# Forward: n_features*l1 + l1*l2 + l2*n_output (MAC per sample)
# Backward: ~2x forward
# epochs × n_train × 3 × (n_features*l1 + l1*l2 + l2*n_output)
max_epochs = 2000
flops_per_sample_forward = n_features * l1 + l1 * l2 + l2 * n_output
flops_train = max_epochs * n_train * 3 * flops_per_sample_forward

# Inference FLOPs
flops_infer_per_sample = flops_per_sample_forward + l1 + l2 + n_output  # MAC + activation
flops_infer = flops_infer_per_sample * n_test_samples

print(f"\n  【B365 Baseline】")
print(f"    推理 FLOPs: {flops_b365:,.0f} ({flops_b365:.2e})")
print(f"    每样本: 10 FLOPs")
print(f"\n  【BalanceNet】")
print(f"    网络结构: {n_features} → {l1} → {l2} → {n_output}")
print(f"    训练 FLOPs: {flops_train:,.0f} ({flops_train:.2e})")
print(f"    推理 FLOPs: {flops_infer:,.0f} ({flops_infer:.2e})")
print(f"    每样本推理: {flops_infer_per_sample} FLOPs")
print(f"    推理开销比 (BalanceNet/B365): {flops_infer / max(flops_b365, 1):.1f}x")

# 保存计算开销
cost_summary = pd.DataFrame({
    'Method': ['B365 (Baseline)', 'BalanceNet'],
    'Training FLOPs': [0, flops_train],
    'Inference FLOPs (test)': [flops_b365, flops_infer],
    'Total FLOPs': [flops_b365, flops_train + flops_infer],
    'Per-sample Inference FLOPs': [10, flops_infer_per_sample],
})
cost_summary.to_csv(os.path.join(SAVE_DIR, 'computational_cost_comparison.csv'),
                    index=False, encoding='utf-8-sig')

# 保存到统一目录
flops_save_dir = '/mnt/train-decision-agent/dingzhijie/Trident-Cat code modification/cat/computational_cost'
os.makedirs(flops_save_dir, exist_ok=True)
flops_df = pd.DataFrame({'Method': ['BalanceNet'], 'Per-sample FLOPs': [flops_infer_per_sample]})
flops_df.to_csv(os.path.join(flops_save_dir, 'balancenet_flops.csv'), index=False, encoding='utf-8-sig')
print(f"Per-sample FLOPs saved to: {os.path.join(flops_save_dir, 'balancenet_flops.csv')}")

# ===== 保存完整结果 =====
summary_path = os.path.join(SAVE_DIR, 'results_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write(f"BalanceNet 对比模型结果汇总 — 联赛: {LEAGUE}\n")
    f.write("=" * 60 + "\n\n")

    f.write("【方法说明】\n")
    f.write("  改编自 Ghost Recon Online 'Beyond Skill Rating' 论文中的 BalanceNet\n")
    f.write("  原文: 玩家统计特征 + Embedding → 线性投影 → 队伍向量(求和) → Tanh隐藏层 → Sigmoid\n")
    f.write("  改编: 主/客队统计特征 → StandardScaler → MLP(Tanh, 两层) → Softmax(3分类)\n\n")

    f.write("【模型参数】\n")
    f.write(f"  隐藏层结构: {best_params['hidden_layer_sizes']}\n")
    f.write(f"  激活函数: Tanh\n")
    f.write(f"  Alpha (L2): {best_params['alpha']:.6f}\n")
    f.write(f"  学习率: {best_params['learning_rate_init']:.6f}\n")
    f.write(f"  Batch size: {best_params['batch_size']}\n")
    f.write(f"  特征数: {n_features}\n")
    f.write(f"  特征列: {feature_cols}\n\n")

    f.write("【RPS 结果】\n")
    f.write(f"  B365 RPS:       {rps_b365:.10f}\n")
    f.write(f"  BalanceNet RPS: {rps_model:.10f}\n")
    f.write(f"  ΔRPS (模型-B365): {rps_model - rps_b365:.10f}\n\n")

    f.write("【Accuracy】\n")
    f.write(f"  BalanceNet Accuracy: {accuracy:.6f}\n")
    f.write(f"  预测占比: Win={(y_pred==0).mean():.3f} | Draw={(y_pred==1).mean():.3f} | Loss={(y_pred==2).mean():.3f}\n\n")

    f.write("【统计检验 (BalanceNet vs B365)】\n")
    f.write(f"  配对 T 检验 (单侧): t={t_res.statistic:.4f}, p={t_res.pvalue:.6f}  {sig_t}\n")
    f.write(f"  Wilcoxon signed-rank (单侧): W={w_res.statistic:.0f}, p={w_res.pvalue:.6f}  {sig_w}\n")
    f.write(f"  Mean Δ RPS = {diff_vec.mean():.10f}\n")
    f.write(f"  BalanceNet 更优样本: {(diff_vec<0).sum()}/{len(diff_vec)} ({100*(diff_vec<0).mean():.1f}%)\n\n")

    f.write("【混淆矩阵】\n")
    f.write(cm_df.to_string() + "\n\n")
    f.write("【Precision / Recall / F1】\n")
    f.write(prf_table.to_string() + "\n\n")

    f.write("【分赛季 RPS】\n")
    season_start_w = 0
    for i, df in enumerate(test_dfs):
        n = len(df)
        season_end_w = season_start_w + n
        y_s = y_test[season_start_w:season_end_w]
        p_s = y_prob[season_start_w:season_end_w]
        b_s = b365_probs[season_start_w:season_end_w]

        rps_m = calc_rps(y_s, p_s)
        rps_b = calc_rps(y_s, b_s)

        rps_m_vec = calc_rps_vec(y_s, p_s)
        rps_b_vec = calc_rps_vec(y_s, b_s)
        d = rps_m_vec - rps_b_vec
        t_s, t_p = stats.ttest_rel(rps_m_vec, rps_b_vec, alternative='less')
        w_s, w_p = stats.wilcoxon(rps_m_vec, rps_b_vec, alternative='less')
        sig = '★★★' if w_p < 0.01 else ('★★' if w_p < 0.05 else ('★' if w_p < 0.10 else '×'))

        f.write(f"\n  —— {test_files[i]} ({n} 场) ——\n")
        f.write(f"    B365 RPS:       {rps_b:.10f}\n")
        f.write(f"    BalanceNet RPS: {rps_m:.10f}\n")
        f.write(f"    T检验: t={t_s:.4f}, p={t_p:.6f}\n")
        f.write(f"    Wilcoxon: p={w_p:.6f} {sig}\n")
        f.write(f"    Mean Δ = {d.mean():.10f}, 模型更优: {(d<0).sum()}/{n} ({100*(d<0).mean():.1f}%)\n")

        season_start_w = season_end_w
    f.write("\n")

    f.write("【计算开销】\n")
    f.write(f"  B365 推理: {flops_b365:,.0f} FLOPs\n")
    f.write(f"  BalanceNet 训练: {flops_train:,.0f} FLOPs\n")
    f.write(f"  BalanceNet 推理: {flops_infer:,.0f} FLOPs\n")
    f.write(f"  每样本推理: B365=10, BalanceNet={flops_infer_per_sample}\n")

print(f"\n{'='*60}")
print("所有结果已保存到:", SAVE_DIR)
print(f"{'='*60}")
