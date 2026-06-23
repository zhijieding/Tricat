import os
import pandas as pd
#一、数据处理（1.合并数据 2. 删除指定列含有空格的行）

#1.合并数据
# 数据文件夹路径
folder_path = r'C:\Users\12197\Desktop\其他赛季\赛季数据'
output_folder = r'C:\Users\12197\Desktop\其他赛季'


# 训练集和测试集文件名
train_files = [
    '1415.csv', '1516.csv', '1617.csv', '1718.csv', '1819.csv',
    '1920.csv', '2021.csv', '2122.csv', 
]
test_files = ['2223.csv','2324.csv']
columns_to_keep = [
    'Date', 'HomeTeam', 'AwayTeam', 'FTR',
    'Hform', 'Aform', 'Hst', 'ASt', 'HSTKPP', 'ASTKPP',
    'HGKPP', 'AGKPP', 'HCKPP', 'ACKPP', 'HAttack', 'AAttack',
    'HMidField', 'AMidField', 'HDefence', 'ADefense', 'HOverall', 'AOverall',
    'HTDG', 'ATDG', 'HStWeighted', 'AStWeighted', 'FormDifferential',
    'StDifferential', 'STKPP', 'GKPP', 'CKPP', 'RelAttack', 'RelMidField',
    'RelDefense', 'RelOverall', 'GDDifferential', 'StWeightedDifferential','HS','AS','HF','AF','FDifferential','SDifferential'
]
# 遍历train_files，读取每个文件并保存为df***.csv、df****.csv
# 遍历test_files，读取每个文件并保存为df***.csv、df****.csv----给最后一步比较不同公司的RPS使用

# 新建两个文件夹存储-新的测试数据和新的训练数据（每个年度1、FTR列转换2、特征列中只要有NaN的行删除得到的数据）
train_output_folder = r'C:\Users\12197\Desktop\其他赛季\train'
test_output_folder = r'C:\Users\12197\Desktop\其他赛季\test'
#递归创建目录（如果父目录不存在，会自动创建）。exist_ok=True，如果目录已存在，不会抛出 FileExistsError 错误。如果设为 False（默认），目录已存在时会报错。
os.makedirs(train_output_folder, exist_ok=True)
os.makedirs(test_output_folder, exist_ok=True)

# 存储-每个年度新的训练数据（1、FTR列转换2、特征列中只要有NaN的行删除得到的数据）保存训练集文件到 train 文件夹
for file in train_files:
    #file_path = os.path.join(folder_path, file)
    # folder_path: 目录路径（如 "data/images"）
    # file: 文件名（如 "cat.jpg"）
    # os.path.join(): 智能拼接路径，自动处理不同操作系统的路径分隔符
    #结果: 生成完整路径（如 "data/images/cat.jpg"）
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path, encoding='utf-8')
    # FTR列转换
    df['FTR'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    #删除指定列中只要有NaN的行
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
    #删除指定列中只要有NaN的行
    df = df.dropna(subset=columns_to_keep, how='any')
    save_name = f"df{file.split('.')[0]}.csv"
    df.to_csv(os.path.join(test_output_folder, save_name), index=False, encoding='utf-8')

# 读取合并数据
import os
import pandas as pd

# 读取并合并训练数据
train_folder = r'C:\Users\12197\Desktop\其他赛季\train'
train_files = [f for f in os.listdir(train_folder) if f.endswith('.csv')]
train_dfs = []
for file in train_files:
    file_path = os.path.join(train_folder, file)
    df = pd.read_csv(file_path, encoding='utf-8')
    train_dfs.append(df)
train_data = pd.concat(train_dfs, ignore_index=True)
#把遇到的NAN转换为0
train_data = train_data.fillna(0)
train_data.to_csv(r'C:\Users\12197\Desktop\其他赛季\train_data.csv', index=False, encoding='utf-8')



# 读取并合并测试数据
test_folder = r'C:\Users\12197\Desktop\其他赛季\test'
test_files = [f for f in os.listdir(test_folder) if f.endswith('.csv')]
test_dfs = []
for file in test_files:
    file_path = os.path.join(test_folder, file)
    df = pd.read_csv(file_path, encoding='utf-8')
    test_dfs.append(df)
test_data = pd.concat(test_dfs, ignore_index=True)
#把遇到的NAN转换为0
test_data = test_data.fillna(0)
test_data.to_csv(r'C:\Users\12197\Desktop\其他赛季\test_data.csv', index=False, encoding='utf-8')
#二、模型训练
# xgboost预测--基于33个特征
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
#A类  'Hform', 'Aform', 'Hst', 'ASt', 'HSTKPP', 'ASTKPP','HGKPP', 'AGKPP', 'HCKPP', 'ACKPP', 'HAttack', 'AAttack','HMidField', 'AMidField', 'HDefence', 'ADefense', 'HOverall', 'AOverall','HTDG', 'ATDG', 'HStWeighted', 'AStWeighted'
#B类  'FormDifferential', 'StDifferential', 'STKPP', 'GKPP', 'CKPP','RelAttack', 'RelMidField', 'RelDefense', 'RelOverall','GDDifferential', 'StWeightedDifferential'
#全  'Hform', 'Aform', 'Hst', 'ASt', 'HSTKPP', 'ASTKPP','HGKPP', 'AGKPP', 'HCKPP', 'ACKPP', 'HAttack', 'AAttack','HMidField', 'AMidField', 'HDefence', 'ADefense', 'HOverall', 'AOverall', 'HTDG', 'ATDG', 'HStWeighted', 'AStWeighted', 'FormDifferential','StDifferential', 'STKPP', 'GKPP', 'CKPP', 'RelAttack', 'RelMidField','RelDefense', 'RelOverall', 'GDDifferential', 'StWeightedDifferential'
# 指定特征列
feature_cols =  [
    'Hform', 'Aform', 'Hst', 'ASt', 'HSTKPP', 'ASTKPP',
    'HGKPP', 'AGKPP', 'HCKPP', 'ACKPP', 'HAttack', 'AAttack',
    'HMidField', 'AMidField', 'HDefence', 'ADefense', 'HOverall', 'AOverall',
    'HTDG', 'ATDG', 'HStWeighted', 'AStWeighted', 'FormDifferential',
    'StDifferential', 'STKPP', 'GKPP', 'CKPP', 'RelAttack', 'RelMidField',
    'RelDefense', 'RelOverall', 'GDDifferential', 'StWeightedDifferential','HS','AS','HF','AF','FDifferential','SDifferential'
]

# 读取处理后的数据
train_data = pd.read_csv(os.path.join(output_folder, 'train_data.csv'), encoding='utf-8')
test_data = pd.read_csv(os.path.join(output_folder, 'test_data.csv'), encoding='utf-8')

# # 随机上采样-处理数据不平衡
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_train,y_train = ros.fit_resample(train_data[feature_cols],train_data['FTR'])
# X_test,y_test = ros.fit_resample(test_data[feature_cols],test_data['FTR'])

# 构建训练集和测试集
# X_train = train_data[feature_cols]
# y_train = train_data['FTR']
X_test = test_data[feature_cols]
y_test = test_data['FTR']


# --------------------------------------------随机森林---------------------------------

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer

# XGBoost参数网格
param_grid =  {
    'n_estimators': [150,160,170,180,190,200,210, 220,250,260,300],
    'learning_rate': [0.01,0.025,0.05,0.1,0.2,0.3],#典型值范围：0.01 到 0.3
    'max_depth': [3, 5, 7, 9,11,13,15,17,19,21],#典型值范围：3 到 10
    'gamma':[0.01,0.02,0.03,0.04,0.05,0.1,0.2,0.5],
    'reg_lambda': [0.01,0.025,0.05,0.075,0.1,0.2,0.6,0.7]

    # 'subsample': [0.5,0.6,0.7,0.8,0.9,1.0],#典型值范围：0.5 到 1.0
    # 'colsample_bytree': [0.5,0.6,0.7,0.8,0.9,1.0],#典型值范围：0.5 到 1.0
    # 'reg_alpha': [0.01,0.02,0.03,0.04,0.05,0.1,0.2,0.5],
    # 'min_child_weight':[1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50],##典型值范围：1 到 10
    # 'max_depth':[3,5,6,9,12,15,17,25]#典型值范围：3 到 10
}

#计算RPS的值
import numpy as np
# 定义RPS函数（如何计算RPS）
def calc_rps(y_true, y_prob, n_class=None):# 定义RPS函数，y_true为真实标签，y_prob为预测概率，n_class为类别数
    if n_class is None:
        n_class = y_prob.shape[1]#如果未指定类别数，则用预测概率的列数
    num_samples = len(y_true)#样本总数
    rps_sum = 0# 初始化RPS总和
    for i in range(num_samples):# 遍历每个样本
        current_probs = y_prob[i]  # 当前样本的预测概率
        current_outcome_scalar = y_true[i]  # 当前样本的真实标签（整数）
        current_outcome_one_hot = np.zeros(n_class) # 创建全零的one-hot向量
        current_outcome_one_hot[current_outcome_scalar] = 1  # 转为one-hot, # 将真实标签位置置为1
        cum_probs = np.cumsum(current_probs)  # 预测概率的累加和
        cum_outcomes = np.cumsum(current_outcome_one_hot)  # 真实标签的累加和
        sum_rps_sample = 0 # 当前样本RPS累加器
        for j in range(n_class):
            sum_rps_sample += (cum_probs[j] - cum_outcomes[j])**2  # 计算每一类的平方差并累加
        if (n_class) > 0:
            rps_sum += sum_rps_sample / (n_class - 1)# 当前样本RPS归一化后累加到总和
    if num_samples > 0:
        return rps_sum / num_samples  # 返回所有样本的平均RPS
    else:
        return 0

# #网格搜索参数-----准确率（accuracy）版
# # 网格搜索参数-----根据模型在交叉验证中的准确率（accuracy）来评估每组参数的表现，并选择准确率最高的参数组合作为最优参数
# rf = RandomForestClassifier(random_state=42)
# grid_search = GridSearchCV(rf, param_grid, cv=2, n_jobs=-1)
# #用训练集的特征数据 X_train 和标签 y_train 对网格搜索对象 grid_search 进行模型训练和参数调优。它会自动遍历你设定的参数组合，找到使模型在交叉验证中表现最好的参数，并保存最优模型。
# grid_search.fit(X_train, y_train)
# # 输出最优参数
# print("最优参数:", grid_search.best_params_)
# print("网格搜索得到的最优RPS:", grid_search.best_score_)


#网格搜索参数-----RPS版

from sklearn.metrics import make_scorer

def rps_scorer(y_true, y_prob):
    # y_true = np.array(y_true)  
    y_true = np.array(y_true, dtype=int)  # 转为整数类型 # 转为numpy数组，避免索引问题
# 常见原因：
# 1. y_true 里有非整数（比如字符串、浮点数、缺失值等），导致不能作为 one-hot 索引。所以取了int类型
# 2. y_true 里有超出类别范围的值，比如类别只有 0,1,2，但出现了 3 或 -1。
    return calc_rps(y_true, y_prob)

custom_scorer = make_scorer(rps_scorer, response_method="predict_proba", greater_is_better=False)
# -----------------------------------------------------------------------------
# ... existing code ...
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

# 定义贝叶斯优化的目标函数
def xgb_evaluate(n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma, reg_lambda):
    params = {
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        'learning_rate': learning_rate,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'gamma': gamma,
        'reg_lambda': reg_lambda,
        'random_state': 42,
        'eval_metric': 'mlogloss'
    }
    model = XGBClassifier(**params)
    # 使用交叉验证评估模型性能，这里我们使用RPS作为评估指标
    # 注意：BayesianOptimization 默认最大化目标函数，而RPS是越小越好，所以我们返回负RPS
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring=custom_scorer, n_jobs=-1)
    return -scores.mean()

# 定义参数的搜索范围
pbounds = {
    'n_estimators': (50, 200),  # 树的数量
    'max_depth': (3, 10),       # 树的最大深度
    'learning_rate': (0.01, 0.3), # 学习率
    'subsample': (0.6, 1.0),    # 训练样本的子采样比例
    'colsample_bytree': (0.6, 1.0), # 每棵树的特征采样比例
    'gamma': (0, 0.5),          # 最小损失减少，用于在树的叶节点上进行进一步分区
    'reg_lambda': (0.01, 10)     # L2正则化项
}

# 初始化贝叶斯优化器
optimizer = BayesianOptimization(
    f=xgb_evaluate,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

# 运行优化
optimizer.maximize(
    init_points=5,  # 初始随机探索点数
    n_iter=20       # 迭代优化次数
)

print("贝叶斯优化得到的最优参数:", optimizer.max['params'])
print("贝叶斯优化得到的最优RPS:", abs(optimizer.max['target']))
# 用于打印贝叶斯优化过程中找到的最佳RPS（Rank Probability Score）值。
# 具体来说：
# - optimizer.max 是贝叶斯优化器 optimizer 找到的最佳结果，它是一个字典，包含了最佳参数 ( 'params' ) 和对应的目标函数值 ( 'target' )。
# - optimizer.max['target'] 获取的是贝叶斯优化目标函数（ xgb_evaluate ）返回的最佳目标值。在之前的对话中，我们提到 xgb_evaluate 函数返回的是负RPS值（因为贝叶斯优化默认是最大化目标函数，而RPS是越小越好，所以我们最大化负RPS）。
# - abs(...) 函数用于取绝对值，将负RPS值转换回正的RPS值，以便于理解和展示。
# 因此，这行代码的输出是贝叶斯优化在训练集上找到的最小RPS值（即最优RPS）。

# 使用最优参数训练最终模型
best_params = optimizer.max['params']
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])

print("贝叶斯优化得到的最优参数:", best_params)
# 使用最优模型在测试集上进行预测
best_xgb = XGBClassifier(**best_params, eval_metric='mlogloss', random_state=42)
best_xgb.fit(X_train, y_train)

# ... existing code ...
# ... existing code ...

# 打印贝叶斯优化得到的最优参数和最优RPS
# print("贝叶斯优化得到的最优参数:", optimizer.max['params'])
print("贝叶斯优化得到的最优RPS:", abs(optimizer.max['target']))

print("贝叶斯优化得到的最优参数:", best_params)
y_prob = best_xgb.predict_proba(X_test)
rps_score_test= calc_rps(y_test.values.astype(int), y_prob)
print("测试集RPS分数:", rps_score_test)

y_pred = best_xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("测试集Accuracy:", accuracy)

# ... existing code ...

# -----------------------------------------------------------------------------
# xgb = XGBClassifier(random_state=42, eval_metric='mlogloss')
# # use_label_encoder=False 表示不使用 XGBoost 自带的标签编码器（新版 XGBoost 推荐直接用 sklearn 的标签编码）。
# # eval_metric='mlogloss' 指定评估指标为多分类对数损失（mlogloss），适用于多分类任务。
# # rf = RandomForestClassifier(random_state=42,class_weight='balanced')
# #根据 RPS（Ranked Probability Score）来进行参数搜索，需要自定义一个评分函数，并通过 scoring 参数传递给 GridSearchCV 。
# #你可以使用 make_scorer 包装你的 calc_rps 函数，并设置 greater_is_better=False （因为RPS越低越好）
# #needs_proba=True 表示评分函数需要模型的预测概率（而不是类别标签），适用于像 RPS 这种基于概率的指标。
# #greater_is_better=False 表示分数越低越好（因为 RPS 越小模型越优），这样网格搜索会自动选择 RPS 最小的参数组合作为最优参数。
# custom_scorer = make_scorer(rps_scorer, response_method="predict_proba", greater_is_better=False)
# #- make_scorer 是 sklearn 用于将自定义函数包装成评分器（scorer），以便 GridSearchCV 等模型选择工具调用。
# # rps_scorer 是你自定义的评分函数（如 calc_rps 或类似函数）。
# # response_method="predict_proba" 表示在评分时会调用模型的 predict_proba 方法，将概率作为参数传给评分函数。
# # greater_is_better=False 表示分数越小越好（比如 RPS 越小模型越优），这样 GridSearchCV 会自动寻找最小值。
# #保持 greater_is_better=False，输出时取负号 如果你用 greater_is_better=False ，sklearn 会自动对分数取负号用于排序
# grid_search = GridSearchCV(xgb, param_grid, cv=2, n_jobs=-1, scoring=custom_scorer)
# grid_search.fit(X_train, y_train)
# print("最优参数:", grid_search.best_params_)

# print("网格搜索得到的最优RPS:", abs(grid_search.best_score_))
# # print("网格搜索得到的最优RPS:", grid_search.best_score_)


# # 最优模型
# best_xgb = grid_search.best_estimator_
# y_prob = best_xgb.predict_proba(X_test)
# rps_score = calc_rps(y_test.values.astype(int), y_prob)
# print("RPS分数:", rps_score)

# # 得到的是每个测试样本的预测类别标签。
# # 如， y_pred[0]=2 表示第一个测试样本被预测为“客场胜”。
# y_pred = best_xgb.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Best Params:", grid_search.best_params_)
# print("Accuracy:", accuracy)
# 如果需要正常显示中文，可以设置 matplotlib 字体为支持中文的字体，例如 SimHei：
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# # # 网格搜索参数-----RPS版调参过程的折线图，热力图（1. n_estimators 与 RPS 折线图 2. max_depth 和 min_samples_split 与 RPS 的热力图）
# # 两个图未匹配文章
# import matplotlib.pyplot as plt
# import pandas as pd

# # 1. n_estimators 与 RPS 折线图
# print("n_estimators 与 RPS 折线图")
# cv_results = pd.DataFrame(grid_search.cv_results_)
# print(cv_results.head())
# #将 GridSearchCV 搜索得到的所有参数组合及其交叉验证结果（字典格式）转换为 pandas 的 DataFrame，方便后续分析和可视化。
# rps_scores = []
# for i in range(len(cv_results)):
#     #cv_results.loc[i, 'params'] 表示取出第 i 行的 params 列内容。
#     params = cv_results.loc[i, 'params']
#     #解包参数
#     xgb = XGBClassifier(random_state=42, **params)
#     xgb.fit(X_train, y_train)
#     y_prob = xgb.predict_proba(X_train)
#     rps = calc_rps(y_train.values.astype(int), y_prob)
#     rps_scores.append(rps)
# cv_results['mean_train_rps'] = rps_scores

# # 画图 -出问题有问题
# plt.figure(figsize=(8,6))
# print("n_estimators 与 RPS 折线图(训练集)")
# plot_df = cv_results.groupby('param_n_estimators')['mean_train_rps'].mean().reset_index()
# plt.plot(plot_df['param_n_estimators'], plot_df['mean_train_rps'], marker='o', markersize=8, markerfacecolor='#FF7F0E', markeredgewidth=0, linewidth=3, color='#FF7F0E', label='RPS分数(训练集)')
# plt.xlabel('决策树数量 n_estimators', fontsize=16, fontweight='bold')
# plt.ylabel('Ranked Probability Score', fontsize=16, fontweight='bold')
# plt.title('n_estimators 与 RPS 折线图(训练集)', fontsize=18, fontweight='bold', loc='center')
# plt.grid(axis='y', linestyle='--', color='#cccccc', alpha=0.8)
# plt.legend(fontsize=15, loc='upper right', frameon=False)
# plt.xticks(fontsize=14, fontweight='bold')
# plt.yticks(fontsize=14, fontweight='bold')
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_linewidth(2)
# plt.gca().spines['bottom'].set_linewidth(2)
# plt.tight_layout()
# plt.savefig(r'C:\Users\12197\Desktop\其他赛季\xgboost\ n_estimators_vs_rps_train.png', dpi=300, bbox_inches='tight')
# plt.show()


# # 2. learning_rate 和 gamma 与 RPS 的热力图
# import seaborn as sns

# print("learning_rate 和 gamma 与 RPS 的热力图(训练集)")
# heatmap_df = cv_results.groupby(['param_gamma','param_learning_rate'])['mean_train_rps'].mean().unstack()
# plt.figure(figsize=(10,7))
# sns.heatmap(heatmap_df, annot=True, fmt='.4f', cmap='RdPu', linewidths=0.5, cbar_kws={'label': 'RPS'})
# plt.xlabel('learning_rate', fontsize=13)
# plt.ylabel('gamma', fontsize=13)
# plt.title('learning_rate 和 gamma 与 RPS 的热力图(训练集)', fontsize=15, fontweight='bold')
# plt.tight_layout()
# plt.savefig(r'C:\Users\12197\Desktop\其他赛季\xgboost\ learning_rate_gamma_rps_heatmap_train.png', dpi=300)
# plt.show()


# #2. reg_lambda 和 max_depth 与 RPS 的热力图
# import seaborn as sns

# print("reg_lambda 和 max_depth 与 RPS 的热力图(训练集)")
# heatmap_df = cv_results.groupby(['param_reg_lambda','param_max_depth'])['mean_train_rps'].mean().unstack()
# plt.figure(figsize=(10,7))
# sns.heatmap(heatmap_df, annot=True, fmt='.4f', cmap='RdPu', linewidths=0.5, cbar_kws={'label': 'RPS'})
# plt.ylabel('reg_lambda', fontsize=13)
# plt.xlabel('max_depth', fontsize=13)
# plt.title('reg_lambda 和 max_depth 与 RPS 的热力图(训练集)', fontsize=15, fontweight='bold')
# plt.tight_layout()
# plt.savefig(r'C:\Users\12197\Desktop\其他赛季\xgboost\ reg_lambda_max_depth_rps_heatmap_train.png', dpi=300)
# plt.show()


#5.混淆矩阵
from sklearn.metrics import confusion_matrix

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)

import pandas as pd

labels = ['Home wins', 'Draws', 'Away wins']  #改了
cm_df = pd.DataFrame(cm, index=labels, columns=['Predicted win', 'Predicted draw', 'Predicted loss'])#改了
print("(a) confusion_matrix")
print(cm_df)

# Precision-Recall Table
from sklearn.metrics import precision_recall_fscore_support

# 计算每一类的precision, recall, f1-score 
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0,1,2])

# 构建DataFrame
prf_table = pd.DataFrame({
    'Precision': precision,
    'Recall': recall,
    'F1-score': f1
}, index=['Home wins', 'Draws', 'Away wins']).applymap(lambda x: round(x, 2)) #改了

print("(b) Precision-recall table")
print(prf_table)

#混淆矩阵和precision, recall, f1-score输出在一张表，输出路径C:\Users\12197\Desktop\traedata\随机森林
# 构造分隔行
sep1 = pd.DataFrame([[""]*3], columns=cm_df.columns, index=[""])
sep2 = pd.DataFrame([[""]*3], columns=prf_table.columns, index=[""])

# 拼接表格
upper = pd.concat([cm_df, sep1], axis=0)
lower = pd.concat([sep2, prf_table], axis=0)
result_table = pd.concat([upper, lower], axis=1)

# 添加多级表头
result_table.columns = pd.MultiIndex.from_tuples(
    [("(a) Confusion matrix", c) if c in cm_df.columns else ("", c) for c in result_table.columns[:3]] +
    [("(b) Precision-recall table", c) if c in prf_table.columns else ("", c) for c in result_table.columns[3:]]
)

# 输出到指定路径
output_path = r'C:\Users\12197\Desktop\模型图\xgboost\result_table.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
result_table.to_csv(output_path, encoding='utf-8-sig')

print("混淆矩阵和Precision/Recall/F1-score已分块输出到：", output_path)
print(result_table)
#shap-可解释机器学习
import matplotlib.pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import shap

#解释器的实例化
explainer = shap.TreeExplainer(best_xgb)
#shap各类条形图
# 计算 SHAP 值
shap_values2 = explainer.shap_values(X_test)
shap.summary_plot(shap_values2, X_test, plot_type="bar")
plt.show()
# # # 绘制 SHAP 摘要图
# # shap.summary_plot(shap_values2[0], X_train, plot_type="bar")
# # plt.show()
# # # 绘制 SHAP 摘要图
# # shap.summary_plot(shap_values2[1], X_train, plot_type="bar")
# # plt.show()
# # # 绘制 SHAP 摘要图
# # shap.summary_plot(shap_values2[2], X_train, plot_type="bar")
# # plt.show()

# # 小提琴图
# shap.summary_plot(shap_values2, X_train, plot_type="violin")
# plt.show()
# # # 绘制 SHAP 摘要图
# # shap.summary_plot(shap_values2[0], X_train)
# # plt.show()
# # # 绘制 SHAP 摘要图
# # shap.summary_plot(shap_values2[1], X_train)
# # plt.show()
# # # 绘制 SHAP 摘要图
# # shap.summary_plot(shap_values2[2], X_train)
# # plt.show()
# # # 计算 SHAP 值-ai备用
# # shap_values = explainer.shap_values(X_test)
# # # 绘制 SHAP 摘要图
# # shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=13)
# # plt.show()
#ROC曲线
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# 假设类别为0,1,2
n_classes = 3
# 将y_test二值化
y_test_bin = label_binarize(y_test, classes=[0,1,2])
y_score = best_xgb.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

#------------------------------------加
# 计算 micro-average ROC curve 和 AUC
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#-----------------------------------------

# --- 应用 Seaborn 风格 ---
plt.style.use('seaborn-v0_8-darkgrid') # 使用深色网格背景风格，非常接近参考图


plt.figure(figsize=(8,6))
colors = ['cyan', 'darkorange', 'cornflowerblue']
labels = ['class H', 'class D', 'class A']
# ------------------加
plt.plot(fpr["micro"], tpr["micro"], 
         label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
         color='deeppink', linestyle=':', linewidth=4)
# -------------------
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of {labels[i]} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=11)
plt.ylabel('True Positive Rate', fontsize=11)
# plt.title('Receiver Operating Curve for Xgboost', fontsize=13)
legend = plt.legend(loc="lower right", fontsize=9, frameon=True, facecolor='white', framealpha=0.8)
# --- 图下方标题 ---
# caption_text = "   (a) ROC curve for Xgboost."
# plt.figtext(0.5, 0.01, caption_text, wrap=True, horizontalalignment='center', fontsize=11)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 为顶部标题和底部figtext调整布局
plt.savefig(r'C:\Users\12197\Desktop\模型图\xgboost\roc_curve.png', dpi=300)
plt.show()

#图10 分类真实结果和预测结果的比较
import matplotlib.pyplot as plt
import numpy as np

# 假设 y_test 是实际结果，y_pred 是预测结果
labels = ['Home Wins', 'Away Wins', 'Draws']
actual_counts = np.bincount(y_test, minlength=3)
predicted_counts = np.bincount(y_pred.astype(int), minlength=3)

x = np.arange(len(labels))  # 标签位置
width = 0.35  # 柱状图的宽度

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, actual_counts, width, label='Actual', color='skyblue')
rects2 = ax.bar(x + width/2, predicted_counts, width, label='Predicted', color='orange')

# 添加标签、标题和自定义x轴刻度
ax.set_ylabel('Counts')
# ax.set_title('Comparison between Actual and Predicted Results by Class (Xgboost)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# 添加数值标签
ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()
# plt.figtext(0.5, 0.01, "(b) Xgboost", wrap=True, horizontalalignment='center', fontsize=11)
plt.savefig(r'C:\Users\12197\Desktop\模型图\xgboost\分类真实结果和预测结果的比较.png', dpi=300)
plt.show()

# ===== Per-sample FLOPs =====
n_features_xgb = len(feature_cols)
# XGBoost: 每棵树遍历 max_depth 层比较 + 1次叶子值
best_n_estimators_xgb = best_params.get('n_estimators', int(best_params.get('n_estimators', 150)))
best_max_depth_xgb = best_params.get('max_depth', int(best_params.get('max_depth', 7)))
flops_xgb_per_sample = best_n_estimators_xgb * (best_max_depth_xgb + 1)
print(f"\n===== Per-sample FLOPs =====")
print(f"XGBoost Per-sample FLOPs: {flops_xgb_per_sample}")
print(f"  (n_estimators={best_n_estimators_xgb}, max_depth={best_max_depth_xgb})")

flops_save_dir = r'/mnt/train-decision-agent/dingzhijie/Trident-Cat code modification/cat/computational_cost'
os.makedirs(flops_save_dir, exist_ok=True)
flops_df = pd.DataFrame({'Method': ['XGBoost'], 'Per-sample FLOPs': [flops_xgb_per_sample]})
flops_df.to_csv(os.path.join(flops_save_dir, 'xgboost_flops.csv'), index=False, encoding='utf-8-sig')
print(f"Saved to: {os.path.join(flops_save_dir, 'xgboost_flops.csv')}")

# # 概率校准曲线--样式待修改
# from sklearn.calibration import calibration_curve
# import matplotlib.pyplot as plt
# import numpy as np

# n_classes = 3
# class_names = ['主场胜', '平局', '客场胜']
# plt.figure(figsize=(8, 6))
# colors = ['#0072B5', '#BC3C29', '#E18727']

# # 预测概率
# y_prob = best_xgb.predict_proba(X_test)

# for i in range(n_classes):
#     prob_true, prob_pred = calibration_curve((y_test == i).astype(int), y_prob[:, i], n_bins=10, strategy='uniform')
#     plt.plot(prob_pred, prob_true, marker='o', label=f'{class_names[i]}', color=colors[i])

# plt.plot([0, 1], [0, 1], 'k--', label='完美校准')
# plt.xlabel('预测概率')
# plt.ylabel('实际概率')
# plt.title('概率校准曲线（多分类）')
# plt.legend()
# plt.tight_layout()
# plt.savefig(r'C:\Users\12197\Desktop\其他赛季\xgboost\calibration_curve.png', dpi=300)
# plt.show()
# 比较RPS值
# import pandas as pd
# import numpy as np

# # # 2. 只保留需要的列
# B365odds_cols = ['B365H', 'B365D', 'B365A']
# PSodds_cols = ['PSH', 'PSD', 'PSA']  
# label_col = 'FTR'

# files = [
#     os.path.join(test_output_folder, 'df2223.csv'),
#     os.path.join(test_output_folder, 'df2324.csv'),
#     os.path.join(output_folder, 'test_data.csv')  # 新增文件
# ]

# #B365博彩公司和Pinnacle Sports--测试的两个年份及合并年份的共3个RPS
# for file in files:
#     #用pandas读取当前文件为DataFrame，编码为utf-8
#     df = pd.read_csv(file, encoding='utf-8')
#     #
#     companies = [
#     (B365odds_cols, 'B365'),
#     (PSodds_cols, 'Pinnacle Sports')
#     ]
#     for odds_cols, company in companies:
#         #删除在指定列（即赔率列 odds_cols 和标签列 label_col ）中只要有一个缺失值（NaN）的所有行，返回一个新的 DataFrame df_valid 。
#         #这样可以确保后续计算 RPS 时，所有用于概率和标签的行数据都是完整的，没有缺失值。
#         df_valid = df.dropna(subset=odds_cols + [label_col], how='any')
#         #提取指定的赔率列（如 ['B365H', 'B365D', 'B365A'] 或 ['PSH', 'PSD', 'PSA'] ），将其转换为 NumPy 数组，并强制类型为 float 。
#         odds = df_valid[odds_cols].values.astype(float)
#         probs = 1 / odds
#         probs = probs / probs.sum(axis=1, keepdims=True)
#        # label_map = {'H': 0, 'D': 1, 'A': 2}
#         #y_true = df_valid[label_col].map(label_map).values.astype(int)
#         y_true = df_valid[label_col].values.astype(int)
#         rps = calc_rps(y_true, probs, n_class=3)
#         print(f"{company}公司 {os.path.basename(file)} 的RPS分数: {rps}")

# #模型--测试的两个年份及合并年份的共3个RPS
# for file in files:
#     df = pd.read_csv(file, encoding='utf-8')
#     # 只保留特征和标签完整的行
#     df_valid = df.dropna(subset=feature_cols + ['FTR'], how='any')
#     X = df_valid[feature_cols]
#     y_true = df_valid['FTR'].values.astype(int)
#     y_prob = best_xgb.predict_proba(X)
#     rps_score_year= calc_rps(y_true, y_prob, n_class=3)
#     print(f"best_xgb模型{os.path.basename(file)} 的RPS分数: {rps_score_year}")
