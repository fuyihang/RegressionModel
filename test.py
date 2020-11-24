
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1.读取数据集
filename = '回归分析.xlsx'
sheet = '销售额'
df = pd.read_excel(filename, sheet)

# 2.特征工程
cols = ['办公费用', '营销费用']
target = '销售额'

X = df[cols]
y = df[target]

# 3.训练模型
# 1)自定义一个模型函数: # 销售额=a*办公费用+b*营销费用^2+c*营销费用+d
def myFun(params, X):
    a,b,c,d = params
    yhat = a*X['办公费用'] + b*X['营销费用']**2 + c*X['营销费用']+d
    return yhat

# 2)定义损失函数
from sklearn.metrics import mean_squared_error
def err(params, *arg):
    # arg = (fun, X, y)
    myf, X, y = arg
    pred = myf(params, X)
    mse = mean_squared_error(y, pred)
    return mse

# 3)求解最优参数
import scipy.optimize as spo

x0 = [1,0.2,0.3,5]  #参数初始值，根据经验值赋值
optResult = spo.minimize(err, x0, 
        args=(myFun, X, y),
        method='trust-constr')
# print(optResult)
if optResult['success']:
    # print('训练成功')
    bestParams = optResult['x']
    print(np.round(bestParams,4))
else:
    print('训练失败！')

# 4.评估模型

