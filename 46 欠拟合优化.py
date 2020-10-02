#-*- coding: utf-8 -*-

########  本文件实现欠拟合回归模型优化，包括
# Part1、去除预测离群值
# Part2、增加非线性检验
# Part3、增加相互作用检验
# Part4、多项式回归
######################################################################

import pandas as pd
import numpy as np

from common import displayRegressionMetrics
from common import intFeatureSelection


# 1.读取数据
filename = '回归分析.xlsx'
sheet = '广告价格与销量'
df = pd.read_excel(filename, sheet_name=sheet)
# print(df.columns.tolist())

# 2、特征选择
cols = ['价格', '广告费用']
target = '销量'

# 属性选择
cols = intFeatureSelection(df, cols, target)

X = df[cols]
y = df[target]

# 3、训练模型
from sklearn.linear_model import LinearRegression
mdl = LinearRegression()
mdl.fit(X, y)

# 4、评估模型
y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)
# 原始R2=0.8337, adjR2=0.8317

######################################################################
########  Part1、去除预测离群值
######################################################################

# 5、优化一：去除预测离群值

# 1）查找预测离群值：3个标准差之外的数据
resid = y - y_pred
std = np.std(resid)
cond = np.abs(resid) > 3*std
dfOutlier = df[cond]
print('预测离群值样本有：\n',dfOutlier)

# 2）去除预测离群值
df.drop(index=dfOutlier.index, inplace=True)

# 3）再回归
X = df[cols]
y = df[target]

mdl = LinearRegression()
mdl.fit(X, y)

# 4）再评估
y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)

# 此时R2=0.8479, adjR2=0.846
# 模型质量由0.83上升到0.85

######################################################################
########  Part2、增加非线性检验
######################################################################

# 5、优化二：非线性检验

# 1）派生非线性变量
df['价格平方'] = df['价格']**2
df['广告平方'] = df['广告费用']**2
cols = df.columns.tolist()
cols.remove(target)

# 2）特征选择
cols = intFeatureSelection(df, cols, target)

X = df[cols]
y = df[target]

# 3）再回归
mdl = LinearRegression()
mdl.fit(X, y)

# 4）再评估
y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)

# 此时R2=0.8917, adjR2=0.8889
# 模型质量由0.85上升到0.89

######################################################################
########  Part3、增加相互作用检验
######################################################################

# 5、优化三：相互作用检验

# 1）派生交互项
df['价格*广告'] = df['价格'] * df['广告费用']
cols = df.columns.tolist()
cols.remove(target)

# 2）特征选择
cols = intFeatureSelection(df, cols, target)

X = df[cols]
y = df[target]

# 3）再回归
mdl = LinearRegression()
mdl.fit(X, y)

# 4）再评估
y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)

# 此时R2=0.9944, adjR2=0.9942
# 模型质量由0.89上升到0.99


######################################################################
########  Part4、多项式回归
######################################################################
# 前面Part2/Part3合并起来，相当于多项式回归


# 1.读取数据
filename = '回归分析.xlsx'
sheet = '身高年龄与体重'
df = pd.read_excel(filename, sheet_name=sheet)
# print(df.columns.tolist())

cols = ['身高', '年龄']
target = '体重'

# 2、属性选择

cols = intFeatureSelection(df, cols, target)

X = df[cols]
y = df[target]

# 3、线性回归
mdl = LinearRegression()
mdl.fit(X, y)

# 4、评估模型
y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)
# R2=0.68, adjR2=0.61

# 5、优化模型（优化数据、新增变量）
from sklearn.preprocessing import PolynomialFeatures

n = 3    #表示最高n次方
po = PolynomialFeatures(degree=n, include_bias=False)
fit_X = po.fit_transform(X)
print(fit_X[:5])

# 3.训练模型
mdl = LinearRegression()
mdl.fit(fit_X, y)

sr = pd.Series(name='回归系数',
        data=[mdl.intercept_]+mdl.coef_.tolist(),
        index=['常数']+list(range(1,fit_X.shape[1]+1)) )
print(sr)

# 4.评估模型(略)
y_pred = mdl.predict(fit_X)
displayRegressionMetrics(y, y_pred, fit_X.shape)
# n=3时，R2=0.875, adjR2=0.314

# 当n=4时，出现R2=1.0，说明过拟合

# 6.应用模型
x = [[151, 12]]  #新数据集

# 先进行多项式转换
fit_X = po.transform(x)

# 再预测
pred = mdl.predict(fit_X) 
print(pred)

