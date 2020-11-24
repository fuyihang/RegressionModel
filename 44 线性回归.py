#-*- coding: utf-8 -*-
# conda install scikit-learn

########  本文件实现回归预测模型，包括
# Part1、一元线性回归
# Part2、多元线性回归
# Part3、线性回归（带类别变量的线性回归）
######################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from common import displayRegressionMetrics

######################################################################
########  Part1、一元线性回归
######################################################################
# 销售额 = k*营销费用 + b


# 1、准备数据集
filename = '回归分析.xlsx'
sheet = '销售额'
df = pd.read_excel(filename, sheet)
# print(df.columns.tolist())

# 2、数据预处理

# 1）数据选取
X = df[['营销费用']]
y = df['销售额']

# 2）可视化观察
plt.figure()
plt.title('营销费用vs销售额')
plt.xlabel('营销费用')
plt.ylabel('销售额')
plt.scatter(X, y)
plt.show()

# 3、训练模型
from sklearn.linear_model import LinearRegression

# 1）训练
mdl = LinearRegression()
mdl.fit(X, y)

# 2）打印回归系数：
print('回归常数b：', mdl.intercept_)
print('回归系数k：', mdl.coef_.tolist())

sr = pd.Series(name='线性回归LR',
        data=[mdl.intercept_]+mdl.coef_.tolist(),
        index=['常数']+X.columns.tolist() )
print(sr)

# 4、评估模型
# 1）模型质量评估指标
R2 = mdl.score(X, y)    #默认的得分，回归中与R2是相等
print('R^2 = ', R2)

#另一种计算形式
# y_pred = mdl.predict(X)
# from sklearn import metrics
# R2 = metrics.r2_score(y, y_pred)
# print('R^2 = ', R2)

# 2）预测值误差评估指标
from sklearn import metrics

y_pred = mdl.predict(X)
MAPE = (abs((y - y_pred)/y)).mean()
MAE = metrics.mean_absolute_error(y, y_pred)
MSE = metrics.mean_squared_error(y, y_pred)
RMSE = MSE**0.5
print('误差评估指标：\nMAPE={},\nMAE={},\nRMSE={}'.format(MAPE, MAE, RMSE))

# 详细参考封装函数：displayRegressionMetrics(y, y_pred,X.shape)

# 5、优化（略）

# 6、应用模型
# 1）预测历史值
y_pred = mdl.predict(X)

# 2）可视化观察
plt.scatter(X, y)               # 画原始数据散点图
plt.plot(X, y_pred, color='k')  #画回归直线

plt.title('营销费用vs销售额')
plt.xlabel('营销费用')
plt.ylabel('销售额')

# 显示回归方程的公式
b = sr[0]        #sr['常数']
k = sr[1]        #sr['营销费用']
formula = '$f(x) = {:.2f}x + {:.0f}$'.format(k, b)
plt.text(7000,11000, formula, fontsize=18)

plt.show()

# 3）保存模型
import joblib
filename = 'out.mdl'
joblib.dump(mdl, filename)

# 4）加载模型
mdlReg = joblib.load(filename)

# 5）预测-新值
XX = [[8000],[9000]]
pred = mdlReg.predict(XX)
print(pred)


######################################################################
########  Part2、多元线性回归
######################################################################
# 销售额 = a*办公费用 + b*营销费用

# 1、读取数据（同上，略）
filename = '回归分析.xlsx'
sheet = '销售额'
df = pd.read_excel(filename, sheet)
# print(df.columns)

# 2、数据预处理

# 1）特征标识
cols = ['办公费用', '营销费用']
target = '销售额'

# 2）特征选择
df2 = df[cols+[target]].corr(method='spearman')
cond = (df2[target] > 0.3)      #假定阈值为0.3
cols = df2[cond].index.tolist()
cols.remove(target)

# 3)构建X, y
X = df[cols]
y = df[target]

# 3、训练模型
mdl = LinearRegression()
mdl.fit(X, y)

sr = pd.Series(name='回归系数',
        data=[mdl.intercept_]+mdl.coef_.tolist(),
        index=['常数']+cols )
print(sr)

# 4、评估模型
print('R^2=', mdl.score(X, y))

y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred,X.shape)

# 5、模型优化（略）
# 6、应用模型（同上，略）


########练习：建立身高、年龄与体重的多元回归模型#######

filename = '回归分析.xlsx'
sheet = '身高年龄与体重'
df = pd.read_excel(filename, sheet)
# print(df.columns.tolist())


######################################################################
########  Part3、带类别自变量的线性回归
######################################################################
# 如果自变量为类别变量，需要转化成哑变量
# 转换原则：变量取k个值，使用k-1个哑变量

# 1、读入数据
filename = '回归分析.xlsx'
sheet = '汽车季度销量'
df = pd.read_excel(filename, sheet_name=sheet)
# print(df.columns.tolist())

# 2、数据处理
# 1)标识属性
catCols = ['季度']
intCols = ['GNP','失业率','利率']
target = '销量'

# 2）特征选择(协方差分析)
import statsmodels.formula.api as smf
import statsmodels.stats.anova as sms

cols = catCols + intCols
formula = '{} ~ {}'.format(
        target,
        '+'.join(cols)
)
module = smf.ols(formula, df).fit()
dfanova = sms.anova_lm(module)

cond = dfanova['PR(>F)'] < 0.05
cols = dfanova[cond].index.tolist()
print('显著影响的因素：', cols)

# 去除无显著影响的因素
for col in catCols:
    if col not in cols:
        catCols.remove(col)
for col in intCols:
    if col not in cols:
        intCols.remove(col)

# # 3）如果要预测，则需要错位/移位，并去除首行
# shiftCols = ['GNP','失业率','利率']
# df[shiftCols] = df[shiftCols].shift(periods = 1, axis=0)
# df.drop(0,axis=0,inplace=True)

# 4）哑变量转换
from sklearn.preprocessing import OneHotEncoder

# 自动找first第一个为默认值
enc = OneHotEncoder(drop='first',sparse=False, dtype='int')
X_ = enc.fit_transform(df[catCols])
# print(X_)

# # 可以手工指定默认值(第四季度)，不过太麻烦，不建议
# labels = (['第四季度', '第一季度','第二季度','第三季度'],)
# enc = OneHotEncoder(categories=labels, drop='first',sparse=False)

# enc.categories_属性中，只有一个数组
cols = []
for cats in enc.categories_:
        cols.extend(cats[1:])
dfCats = pd.DataFrame(
        data = X_, 
        index=df.index, 
        columns=cols)
# print(dfCats.head())

# 4）合并连续变量和哑变量
dfCols = pd.concat([dfCats, df[intCols] ], axis=1)
# print(dfCols.head())
cols = dfCols.columns.tolist()

# 5)构造X,y
X = dfCols
y = df[target]

# 3、训练模型
mdl = LinearRegression()
mdl.fit(X, y)

sr = pd.Series(name='回归系数',
            data=[mdl.intercept_] + mdl.coef_.tolist(),
            index=['常数']+ cols)
print(sr)

# 4、评估模型
y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)

# 5、模型优化（略）
# 6、模型应用
# 预测87年第一季节的汽车销量
X0 = [['第一季度', 4716, 6.9, 7.4]]
dfX0 = pd.DataFrame(
        data=X0,
        columns=catCols+intCols)
print(dfX0)

# 先哑变量化，再合并
X_ = enc.transform(dfX0[catCols])

X = np.concatenate([X_, dfX0[intCols]],axis=1)
pred = mdl.predict(X)
print('预测销量：',pred)


############################
######## 练习、带类别自变量的线性回归
############################
# 查看员工性别、工龄对终端销量的影响

# 1、准备数据
filename = '回归分析.xlsx'
sheet = '终端销量'
df = pd.read_excel(filename, sheet_name=sheet)

catCols = ['性别']
intCols = ['工龄']
target = '年销量'

# 2、数据预处理

# 1）特征筛选（略）
grouped = df.groupby('性别')
sr = grouped[target].mean()
sr.plot(kind='bar', title='平均销量')

# 2）类别型变量进行哑变量化
enc = OneHotEncoder(drop='first', sparse=False)
X_ = enc.fit_transform(df[catCols])
print(X_[:5])

dfCats = pd.DataFrame(X_, 
            index=df.index, 
            columns=enc.categories_[0][1:])
print(dfCats.head())

# 3）合并数据
dfInts = df[intCols]

dfCols = pd.concat([dfInts, dfCats], axis=1)
print(dfCols.head())

cols = dfCols.columns.tolist()
X = dfCols
y = df[target]

# 3、训练模型
mdl = LinearRegression()
mdl.fit(X, y)

sr = pd.Series(name='销量',
            data=[mdl.intercept_] + mdl.coef_.tolist(),
            index=['常数']+ cols)
print(sr)

# 4、评估模型
y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)

# 5、优化略
# 6、应用模型

# 假定待预测的数据集
data = [['男', 24], ['女',48]]

df2 = pd.DataFrame(data, columns=['性别','工龄'])

# 转换成哑变量
X_ = enc.transform(df2[catCols])

# 再合并
import numpy as np

XX = np.concatenate((df2[intCols], X_), axis=1)
print(XX)

y_pred = mdl.predict(XX)
print(y_pred)

