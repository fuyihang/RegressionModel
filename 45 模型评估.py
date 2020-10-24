#-*- coding: utf-8 -*-

########  本文件主要实现模型评估方法，包括
# Part1、样本集划分
# Part2、留出法Houd-out
# Part3、交叉验证k-fold cross validate
# Part4、交叉验证（手工）
######################################################################

import numpy as np
import pandas as pd

######################################################################
########  Part1、样本集划分
######################################################################

# from sklearn.model_selection import train_test_split

# # 训练集 vs 测试集
# test_size为0~1的小数时，表示百分比，默认为0.25
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# # 训练集 vs 验证集 vs 测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

# 如果使用超参优化，则直接在X_train集中训练和超参优化，顺便将最优模型都得到了

######################################################################
########  Part2、留出法Houd-out，也称简单交叉验证
######################################################################
# 训练集 vs 测试集

# 1.读数据
filename = '回归分析.xlsx'
sheet = '销售额'
df = pd.read_excel(filename, sheet)
# print(df.columns.tolist())

# 2.数据预处理

# 1）特征选择
cols = ['办公费用','营销费用']
target = '销售额'

X = df[cols]
y = df[target]

# 2）样本划分
from sklearn.model_selection import train_test_split

# 训练集 vs 测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 3.训练模型
from sklearn.linear_model import LinearRegression

mdl = LinearRegression()
mdl.fit(X_train, y_train)

# 4.评估模型
from common import displayRegressionMetrics

print('训练集-评估指标')
pred = mdl.predict(X_train)
displayRegressionMetrics(y, pred)

print('测试集-评估指标')
pred = mdl.predict(X_test)
displayRegressionMetrics(y_test, pred)


######################################################################
########  Part3、K折交叉验证（k-fold cross_val_score）
######################################################################

# 4.评估模型
from sklearn.model_selection import cross_val_score

mdl = LinearRegression()

# scoring的取值请参考：sklearn.metrics.SCORERS.keys()
scoring = 'r2'    #'r2'
scores = cross_val_score(mdl, X, y, cv=5, scoring=scoring)
miu = scores.mean()
delt =  scores.std()
print('\n平均值:{0:.2f},标准差:{1:.2f}'.format(miu, delt))
print('置信区间%95：[{:.2f},{:.2f}]'.format(miu-2*delt, miu+2*delt))


######################################################################
########  Part4、交叉验证（手工）
######################################################################
# 下面的评估结果与上面cross_val_score是一样的

from sklearn.model_selection import KFold

scores = []
kf = KFold(n_splits=5, random_state=0)

for train_index, test_index in kf.split(X):
    X_train = X.iloc[train_index, :]
    y_train = y.iloc[train_index]

    X_test = X.iloc[test_index, :]
    y_test = y.iloc[test_index]

    mdl = LinearRegression()
    mdl.fit(X_train, y_train)

    r2 = mdl.score(X_test, y_test)
    scores.append(r2)

# print(scores)
miu = np.mean(scores)
delt =  np.std(scores)
print('\n{0}:平均值:{1:.2f},标准差:{2:.2f}'.format('手工k折', miu,delt))
print('置信区间%95：[{:.2f},{:.2f}]'.format(miu-2*delt, miu+2*delt))

############################
######## 交叉验证：其它扩展的交叉验证
############################
# # 交叉验证的其它类

# from sklearn.model_selection import KFold,LeaveOneOut,LeavePOut,StratifiedKFold
# kf = KFold(n_splits = 3)    #K折交叉
# kf = RepeatedKFold()        #K折重复多次
# kf = LeaveOneout()          #留一交叉
# kf = LeavePOut(p=2)         #留p划分子集
# kf = StratifiedKFold(n_splits=3)    #分层K折交叉

