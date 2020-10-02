#-*- coding: utf-8 -*-
# conda install scikit-learn

########  本文件实现随机梯度下降回归模型，包括
# Part1、随机梯度下降回归SGDRegressor
# Part2、大规律数据集SGD建模

######################################################################
# 随机梯度下降，最大的好处在于，通过迭代解决大规模数据建模

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from common import displayRegressionMetrics

######################################################################
########  Part1、随机梯度下降回归
######################################################################

# 1、读取数据
filename = '回归分析.xlsx'
sheet = '广告价格与销量'
df = pd.read_excel(filename, sheet)
# print(df.columns.tolist())

# 2、特征选择
cols = ['价格', '广告费用']
target = '销量'

X = df[cols]
y = df[target]

# 划分数据集
from sklearn import model_selection as ms

X, X_test, y, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=1)

# 训练集X, y
# 测试集X_test, y_test

# 3、训练模型，及超参优化
from sklearn.linear_model import SGDRegressor

# 固定的参数在初始化类时确定
fixedParams = dict(random_state=10,learning_rate='adaptive',max_iter=10000)
mdl = SGDRegressor(**fixedParams)

# 待优化的超参，由网格搜索确定
params = [{'loss':['squared_loss', 'huber', 'epsilon_insensitive','squared_epsilon_insensitive'],
            'penalty':['l2', 'l1'],
            'alpha':np.linspace(0, 10, 50),         #l1,l2时仅使用alpha超参            
            # 'max_iter':[5000, 10000, 50000]
        },
        {'loss':['squared_loss', 'huber', 'epsilon_insensitive','squared_epsilon_insensitive'],
            'penalty':['elasticnet'],
            'l1_ratio':np.linspace(0, 1, 20),       #范围[0,1],仅用于elasticnet惩罚项
            # 'max_iter':[1000, 5000, 10000],
        }]

# 超参优化
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(mdl, params)
grid.fit(X, y)

print('最优超参：', grid.best_params_)
print('最优得分：',grid.best_score_)

bestParams = fixedParams
bestParams.update(grid.best_params_)

mdl = grid.best_estimator_  #保存最优模型
#奇怪：mdl.intercept_返回的是一个数组，而不是单个值
sr = pd.Series(
        data = mdl.intercept_.tolist() + mdl.coef_.tolist(), 
        index= ['常数'] + cols
)
print(sr)

# 4、评估
# 1)训练集评估
y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)

# 2)测试集评估
y_pred = mdl.predict(X_test)
displayRegressionMetrics(y_test, y_pred, X_test.shape)

# 6、应用模型
XX = [[10, 22]]
pred = mdl.predict(XX)
print(pred)


######################################################################
########  Part2、大规模数据集的随机梯度下降回归 示例
######################################################################
# 每次读取部分数据放入内存（而不是一次性）
# 递归训练模型


filename = '回归分析.xlsx'
sheet = '广告价格与销量'

cols = ['价格', '广告费用']
target = '销量'

from sklearn.linear_model import SGDRegressor

# 注意：一定要设置参数warm_start=True
mdl = SGDRegressor(random_state=10,learning_rate='adaptive', warm_start=True)

# 每次读部分数据进行训练
batch_size = 20
skiprows = 0

params = dict(
        io=filename, sheet_name = sheet, 
        header = 0,
        names= [target] + cols,     #要加上标题，否则下次读时会把第一行当成标题
        nrows=batch_size
)
df = pd.read_excel(skiprows=skiprows, **params)
while not df.empty:
    X = df[cols]
    y = df[target]

    mdl.fit(X, y)

    skiprows += batch_size
    df = pd.read_excel(skiprows=skiprows,**params)
else:
    sr = pd.Series(
            data=mdl.intercept_.tolist() + mdl.coef_.tolist(),
            index=['常数'] + cols
    )
    print(sr)

print('end')


# SGDRegressor类
    # SGDRegressor(loss='squared_loss', penalty='l2', 
    #           alpha=0.0001, l1_ratio=0.15, fit_intercept=True, 
    #           max_iter=1000, tol=0.001, shuffle=True, verbose=0, 
    #           epsilon=DEFAULT_EPSILON, random_state=None, 
    #           learning_rate='invscaling', eta0=0.01, power_t=0.25, 
    #           early_stopping=False, validation_fraction=0.1, 
    #           n_iter_no_change=5, warm_start=False, average=False)
    # SGDRegressor 非常适用于有大量训练样本（>10,000)的回归问题，
    # 对于其他问题，推荐使用 Ridge ，Lasso ，或 ElasticNet 。

    # SGDRegressor 支持以下的损失函数:
        # loss=”squared_loss”: Ordinary least squares（普通最小二乘法）,
        # loss=”huber”: Huber loss for robust regression（Huber回归）,
        # loss=”epsilon_insensitive”: linear Support Vector Regression（线性支持向量回归）.
    # Huber 和 epsilon-insensitive 损失函数可用于 robust regression（鲁棒回归）。
    # 不敏感区域的宽度必须通过参数 epsilon 来设定。这个参数取决于目标变量的规模。
    # SGDRegressor 支持 ASGD（平均随机梯度下降） 作为 SGDClassifier。均值化可以通过设置 average=True 来启用。

    # SGDRegressor 对于利用了 squared loss（平方损失）和 l2 penalty（l2惩罚）的回归，
    # 在 Ridge 中提供了另一个采取 averaging strategy（平均策略）的 SGD 变体，其使用了随机平均梯度 (SAG) 算法。

#SGDClassifier类
    #SGDClassifier 支持以下的 loss functions（损失函数）：
        # loss=”hinge”: (soft-margin) linear Support Vector Machine （（软-间隔）线性支持向量机），
        # loss=”modified_huber”: smoothed hinge loss （平滑的 hinge 损失），
        # loss=”log”: logistic regression （logistic 回归），
        # and all regression losses below（以及所有的回归损失）。

    # SGDClassifier 支持以下 penalties（惩罚）:
        # penalty=”l2”: L2 norm penalty on coef_.（默认）
        # penalty=”l1”: L1 norm penalty on coef_.
        # penalty=”elasticnet”: Convex combination of L2 and L1（L2 型和 L1 型的凸组合）; 
        #           (1 - l1_ratio) * L2 + l1_ratio * L1.
    # L1 penalty （惩罚）导致稀疏解，使得大多数系数为零。 
    # Elastic Net（弹性网）解决了在特征高相关时 L1 penalty（惩罚）的一些不足。
    # 参数 l1_ratio 控制了 L1 和 L2 penalty（惩罚）的 convex combination （凸组合）。
