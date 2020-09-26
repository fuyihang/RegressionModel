#-*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

######################################################################
########  Part1、自定义模型：无约束条件
######################################################################

# 1.读数据集
filename = '回归分析.xlsx'
sheet = '销售额'
df = pd.read_excel(filename, sheet)


# 2.特征选择
cols = ['办公费用', '营销费用']
target = '销售额'
X = df[cols]
y = df[target]

# 3.训练模型
# 1）自定义模型
def yhat(params, *arg):
    a,b,c,d = params
    X,y = arg

    ypred = a*X['办公费用'] + b*X['营销费用']**2 + c*X['营销费用'] + d
    return ypred

# 2）自定义损失函数
import sklearn.metrics as mts
def error(params, *arg):
    pred = yhat(params, *arg)

    mse = mts.mean_squared_error(y, pred)
    return mse

# 3）最优化参数
import scipy.optimize as spo

params = [10, 20, 3, 4]   #初始值
optResult = spo.minimize(error, params, args=(X, y) )
# print(optResult)

bestParams = optResult['x']
print(bestParams)

# 4.评估模型
y_pred = yhat(bestParams, *(X,y))
# print(y_pred)

mae = np.abs(y - y_pred).mean()
mape = np.abs( (y-y_pred)/y ).mean()


######################################################################
########  Part2、自定义模型：带约束条件
######################################################################

# 1.读数据集
filename = '回归分析.xlsx'
sheet = '航空里程'
df = pd.read_excel(filename, sheet)

# 2.特征工程
col = '时间'
df['t'] = range(1, len(df)+1)
df['month'] = df[col].dt.month
df['y'] = df['里程（万）']

X = df[['t','month']]
y = df['y']

# 3.训练模型
def yhat2(params, X):
    base, trend, *sv = params
    svfactor = lambda x: sv[x-1]
    y = base + trend * X['t'] + df['month'].apply(svfactor)
    return y

import sklearn.metrics as mts
def error2(params, *arg):
    X, y = arg
    pred = yhat2(params, X)
    mse = mts.mean_squared_error(y, pred)
    return mse


# 初始值
params = [30, 1, 1,2,3,4,5,6,7,8,9,10,1,2]

# 约束条件
# type='eq',表示fun值=0；type='ineq',表示fun值为非负数
cons = ({'type':'eq', 'fun': lambda params: params[2:].sum()})
# 边界
sv = [(-100,100)]*12      #季节因子的取值范围
bnds = [(30,50),(-5,5)]  #其它因子的取值范围
bnds.extend(sv)

import scipy.optimize as spo
optResult = spo.minimize(error2, params, args=(X, y), 
            constraints=cons,
            bounds=bnds
            )
bestParams = optResult['x']

sr = pd.Series(
        data=bestParams,
        index=['base','trend','m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12']
)
print(sr)

pred = yhat2(bestParams, X)

plt.plot(df.index, df['里程（万）'], label='里程(万)')
plt.plot(df.index, pred, label='预测值')
plt.legend()
plt.show()

# 4.评估模型
mape = np.abs( (y-pred)/y ).mean()
print('MAPE={:.2%}'.format(mape))
