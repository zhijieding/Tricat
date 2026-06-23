# -*- coding: utf-8 -*-
import os
import pandas as pd

# 一、数据处理（1.合并数据 2. 删除指定列含有空格的行）

# 1.合并数据
# 数据文件夹路径
folder_path = r'C:\Users\86152\pythonProject\pythonProject\Trident-Cat code modification\数据\赛季数据'
output_folder = r'C:\Users\86152\pythonProject\pythonProject\Trident-Cat code modification\数据'

# 训练集和测试集文件名
train_files = [
    '1415.csv', '1516.csv', '1617.csv', '1718.csv', '1819.csv',
    '1920.csv', '2021.csv', '2122.csv',
]
test_files = ['2223.csv', '2324.csv']
columns_to_keep = [
    'Date', 'HomeTeam', 'AwayTeam', 'FTR',
    'Hform', 'Aform', 'Hst', 'ASt', 'HSTKPP', 'ASTKPP',
    'HGKPP', 'AGKPP', 'HCKPP', 'ACKPP', 'HAttack', 'AAttack',
    'HMidField', 'AMidField', 'HDefence', 'ADefense', 'HOverall', 'AOverall',
    'HTDG', 'ATDG', 'HStWeighted', 'AStWeighted', 'FormDifferential',
    'StDifferential', 'STKPP', 'GKPP', 'CKPP', 'RelAttack', 'RelMidField',
    'RelDefense', 'RelOverall', 'GDDifferential', 'StWeightedDifferential', 'HS', 'AS', 'HF', 'AF', 'FDifferential',
    'SDifferential'
]
# 遍历train_files，读取每个文件并保存为df***.csv、df****.csv
# 遍历test_files，读取每个文件并保存为df***.csv、df****.csv----给最后一步比较不同公司的RPS使用

# 新建两个文件夹存储-新的测试数据和新的训练数据（每个年度1、FTR列转换2、特征列中只要有NaN的行删除得到的数据）
train_output_folder = r'C:\Users\86152\pythonProject\pythonProject\Trident-Cat code modification\数据\train'
test_output_folder = r'C:\Users\86152\pythonProject\pythonProject\Trident-Cat code modification\数据\test'
# 递归创建目录（如果父目录不存在，会自动创建）。exist_ok=True，如果目录已存在，不会抛出 FileExistsError 错误。如果设为 False（默认），目录已存在时会报错。
os.makedirs(train_output_folder, exist_ok=True)
os.makedirs(test_output_folder, exist_ok=True)

# 存储-每个年度新的训练数据（1、FTR列转换2、特征列中只要有NaN的行删除得到的数据）保存训练集文件到 train 文件夹
for file in train_files:
    # file_path = os.path.join(folder_path, file)
    # folder_path: 目录路径（如 "data/images"）
    # file: 文件名（如 "cat.jpg"）
    # os.path.join(): 智能拼接路径，自动处理不同操作系统的路径分隔符
    # 结果: 生成完整路径（如 "data/images/cat.jpg"）
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path, encoding='utf-8')
    # FTR列转换
    df['FTR'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    # 删除指定列中只要有NaN的行
    df = df.dropna(subset=columns_to_keep, how='any')
    # 保存处理后的文件，文件名分别为df1415.csv、df1516.csv
    save_name = f"df{file.split('.')[0]}.csv"
    df.to_csv(os.path.join(train_output_folder, save_name), index=False, encoding='utf-8')

# 存储-每个年度新的测试数据（1、FTR列转换2、特征列中只要有NaN的行删除得到的数据）保存训练集文件到  test 文件夹
for file in test_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path, encoding='utf-8')
    # FTR列转换
    df['FTR'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    # 删除指定列中只要有NaN的行
    df = df.dropna(subset=columns_to_keep, how='any')
    save_name = f"df{file.split('.')[0]}.csv"
    df.to_csv(os.path.join(test_output_folder, save_name), index=False, encoding='utf-8')

# 读取合并数据
import os
import pandas as pd

# 读取并合并训练数据
train_folder = r'C:\Users\86152\pythonProject\pythonProject\Trident-Cat code modification\数据\train'
train_files = [f for f in os.listdir(train_folder) if f.endswith('.csv')]
train_dfs = []
for file in train_files:
    file_path = os.path.join(train_folder, file)
    df = pd.read_csv(file_path, encoding='utf-8')
    train_dfs.append(df)
train_data = pd.concat(train_dfs, ignore_index=True)
# 把遇到的NAN转换为0
train_data = train_data.fillna(0)
train_data.to_csv(r'C:\Users\86152\pythonProject\pythonProject\Trident-Cat code modification\数据\train_data.csv', index=False, encoding='utf-8')

# 读取并合并测试数据
test_folder = r'C:\Users\86152\pythonProject\pythonProject\Trident-Cat code modification\数据\test'
test_files = [f for f in os.listdir(test_folder) if f.endswith('.csv')]
test_dfs = []
for file in test_files:
    file_path = os.path.join(test_folder, file)
    df = pd.read_csv(file_path, encoding='utf-8')
    test_dfs.append(df)
test_data = pd.concat(test_dfs, ignore_index=True)
# 把遇到的NAN转换为0
test_data = test_data.fillna(0)
test_data.to_csv(r'C:\Users\86152\pythonProject\pythonProject\Trident-Cat code modification\数据\test_data.csv', index=False, encoding='utf-8')
# 二、模型训练
# 高斯朴素贝叶斯预测--基于33个特征
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# A类  'Hform', 'Aform', 'Hst', 'ASt', 'HSTKPP', 'ASTKPP','HGKPP', 'AGKPP', 'HCKPP', 'ACKPP', 'HAttack', 'AAttack','HMidField', 'AMidField', 'HDefence', 'ADefense', 'HOverall', 'AOverall','HTDG', 'ATDG', 'HStWeighted', 'AStWeighted'
# B类  'FormDifferential', 'StDifferential', 'STKPP', 'GKPP', 'CKPP','RelAttack', 'RelMidField', 'RelDefense', 'RelOverall','GDDifferential', 'StWeightedDifferential'
# 全  'Hform', 'Aform', 'Hst', 'ASt', 'HSTKPP', 'ASTKPP','HGKPP', 'AGKPP', 'HCKPP', 'ACKPP', 'HAttack', 'AAttack','HMidField', 'AMidField', 'HDefence', 'ADefense', 'HOverall', 'AOverall', 'HTDG', 'ATDG', 'HStWeighted', 'AStWeighted', 'FormDifferential','StDifferential', 'STKPP', 'GKPP', 'CKPP', 'RelAttack', 'RelMidField','RelDefense', 'RelOverall', 'GDDifferential', 'StWeightedDifferential'
# 指定特征列
feature_cols = [
    'Hform', 'Aform', 'Hst', 'ASt', 'HSTKPP', 'ASTKPP',
    'HGKPP', 'AGKPP', 'HCKPP', 'ACKPP', 'HAttack', 'AAttack',
    'HMidField', 'AMidField', 'HDefence', 'ADefense', 'HOverall', 'AOverall',
    'HTDG', 'ATDG', 'HStWeighted', 'AStWeighted', 'FormDifferential',
    'StDifferential', 'STKPP', 'GKPP', 'CKPP', 'RelAttack', 'RelMidField',
    'RelDefense', 'RelOverall', 'GDDifferential', 'StWeightedDifferential', 'HS', 'AS', 'HF', 'AF', 'FDifferential',
    'SDifferential'
]

# 读取处理后的数据
train_data = pd.read_csv(os.path.join(output_folder, 'train_data.csv'), encoding='utf-8')
test_data = pd.read_csv(os.path.join(output_folder, 'test_data.csv'), encoding='utf-8')

# # 随机上采样-处理数据不平衡
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)
X_train, y_train = ros.fit_resample(train_data[feature_cols], train_data['FTR'])
# X_test,y_test = ros.fit_resample(test_data[feature_cols],test_data['FTR'])

# 构建训练集和测试集
# X_train = train_data[feature_cols]
# y_train = train_data['FTR']
X_test = test_data[feature_cols]
y_test = test_data['FTR']

# --------------------------------------------随机森林---------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# 随机森林参数网格（按照文章参数的范围）
# {
#     'n_estimators': list(range(1, 401, 2)),           # 1到400
#     'max_depth': list(range(1, 14)),               # 1到13
#     'min_samples_split': list(range(1, 101)),      # 1到100
#     'criterion': ['gini']
# }
param_grid = {
    'n_estimators': [50, 100, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300],
    # 从1到400，步长为2,
    'max_depth': [8, 9, 10, 11, 12, 13],
    'min_samples_split': [2, 5, 10, 15, 100],
    'criterion': ['gini']
}

# 计算RPS的值
import numpy as np


# 定义RPS函数（如何计算RPS）
def calc_rps(y_true, y_prob, n_class=None):  # 定义RPS函数，y_true为真实标签，y_prob为预测概率，n_class为类别数
    if n_class is None:
        n_class = y_prob.shape[1]  # 如果未指定类别数，则用预测概率的列数
    num_samples = len(y_true)  # 样本总数
    rps_sum = 0  # 初始化RPS总和
    for i in range(num_samples):  # 遍历每个样本
        current_probs = y_prob[i]  # 当前样本的预测概率
        current_outcome_scalar = y_true[i]  # 当前样本的真实标签（整数）
        current_outcome_one_hot = np.zeros(n_class)  # 创建全零的one-hot向量
        current_outcome_one_hot[current_outcome_scalar] = 1  # 转为one-hot, # 将真实标签位置置为1
        cum_probs = np.cumsum(current_probs)  # 预测概率的累加和
        cum_outcomes = np.cumsum(current_outcome_one_hot)  # 真实标签的累加和
        sum_rps_sample = 0  # 当前样本RPS累加器
        for j in range(n_class):
            sum_rps_sample += (cum_probs[j] - cum_outcomes[j]) ** 2  # 计算每一类的平方差并累加
        if (n_class) > 0:
            rps_sum += sum_rps_sample / (n_class - 1)  # 当前样本RPS归一化后累加到总和
    if num_samples > 0:
        return rps_sum / num_samples  # 返回所有样本的平均RPS
    else:
        return 0

# 网格搜索参数-----RPS版

from sklearn.metrics import make_scorer


def rps_scorer(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    return calc_rps(y_true, y_prob)


# 关键：让 scorer 返回“负的 RPS”，这样“越大越好”
neg_rps_scorer = make_scorer(rps_scorer, response_method="predict_proba",
                             greater_is_better=False)

from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline as SkPipeline  # ★修正点：用于CV内放Scaler

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 定义贝叶斯优化的目标函数（ANN版本）
def ann_evaluate(hidden_layer_size, alpha, learning_rate_init, batch_size):  # <-- 修改了输入参数
    """
    ANN的贝叶斯优化目标函数 - 强力正则化版本
    """
    params = {
        'hidden_layer_sizes': (int(hidden_layer_size),),
        'alpha': alpha,
        'learning_rate_init': learning_rate_init,
        'batch_size': int(batch_size),
        'solver': 'adam',
        'activation': 'relu',
        'max_iter': 1500,
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.15,
        'n_iter_no_change': 5,
        'tol': 0.001
    }

    model = MLPClassifier(**params)

    # --- 5. 更稳健的交叉验证 ---
    from sklearn.model_selection import RepeatedKFold
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

    # ★修正点1：用 Pipeline 在 CV 折内拟合 StandardScaler，避免泄漏
    pipe = SkPipeline([('scaler', StandardScaler()), ('clf', model)])

    # ★修正点2：用 neg_rps_scorer（返回“负的 RPS”），且不再取负号
    cv_score = cross_val_score(pipe, X_train, y_train,
                               scoring=neg_rps_scorer, cv=cv, n_jobs=-1).mean()
    return cv_score  # 直接返回（负的 RPS）；Bayes 将最大化它（越接近0越好）


# 定义ANN的参数搜索范围 - 强力正则化版本
pbounds = {
    'hidden_layer_size': (8, 48),
    'alpha': (1.0, 10.0),
    'learning_rate_init': (1e-4, 5e-3),
    'batch_size': (64, 512)
}

# 初始化贝叶斯优化器
optimizer = BayesianOptimization(
    f=ann_evaluate,
    pbounds=pbounds,
    random_state=42
)

optimizer.maximize(init_points=10, n_iter=30)

# 提取并打印最优参数
best_params_raw = optimizer.max['params']
best_params = {
    'hidden_layer_sizes': (int(best_params_raw['hidden_layer_size']),),
    'alpha': best_params_raw['alpha'],
    'learning_rate_init': best_params_raw['learning_rate_init'],
    'batch_size': int(best_params_raw['batch_size']),
    'random_state': 42,
    'max_iter': 2000,
    'early_stopping': True,
    'validation_fraction': 0.15,
    'n_iter_no_change': 5,
    'tol': 0.001
}

print("贝叶斯优化得到的最优参数:", best_params)
print("贝叶斯优化得到的最优RPS:", abs(optimizer.max['target']))

# 使用最优参数训练最终的ANN模型（保持你原本外部缩放）
best_ann = MLPClassifier(**best_params)
best_ann.fit(X_train_scaled, y_train)

# 在测试集上进行预测（保持概率用缩放后的特征）
y_prob = best_ann.predict_proba(X_test_scaled)
rps_score = calc_rps(y_test.values.astype(int), y_prob)
print("测试集RPS分数:", rps_score)

# ★修正点3：分类预测同样用缩放后的特征
y_pred = best_ann.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("测试集Accuracy:", accuracy)

# ===== Per-sample FLOPs =====
n_features_ann = len(feature_cols)
n_classes_ann = 3
# ANN (单隐藏层 MLP): Input -> Hidden(h) -> Output
# Layer1: n_features*h MAC(乘加各1) + h activations(ReLU ~1)
# Layer2: h*n_classes MAC
# Softmax: ~15
h_size = best_params['hidden_layer_sizes'][0]
flops_ann_per_sample = n_features_ann * h_size * 2 + h_size + h_size * n_classes_ann * 2 + 15
print(f"\n===== Per-sample FLOPs =====")
print(f"ANN Per-sample FLOPs: {flops_ann_per_sample}")
print(f"  (hidden_size={h_size}, n_features={n_features_ann})")

flops_save_dir = r'/mnt/train-decision-agent/dingzhijie/Trident-Cat code modification/cat/computational_cost'
os.makedirs(flops_save_dir, exist_ok=True)
flops_df = pd.DataFrame({'Method': ['ANN'], 'Per-sample FLOPs': [flops_ann_per_sample]})
flops_df.to_csv(os.path.join(flops_save_dir, 'ann_flops.csv'), index=False, encoding='utf-8-sig')
print(f"Saved to: {os.path.join(flops_save_dir, 'ann_flops.csv')}")
