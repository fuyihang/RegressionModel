#-*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    a,b,c = params
    X,y = arg

    ypred = a*X['办公费用'] + b*X['营销费用'] + c
    return ypred

# 2）自定义损失函数
import sklearn.metrics as mts
def error(params, *arg):
    pred = yhat(params, *arg)

    mse = mts.mean_squared_error(y, pred)
    return mse

# 3）最优化参数
import scipy.optimize as spo

params = [10,200,300]   #初始值
optResult = spo.minimize(error, params, args=(X, y) )
# print(optResult)

bestParams = optResult['x']
print(bestParams)

# 4.评估模型
y_pred = yhat(bestParams, *(X,y))
print(y_pred)

mae = np.abs(y - y_pred).mean()
mape = np.abs( (y-y_pred)/y ).mean()

