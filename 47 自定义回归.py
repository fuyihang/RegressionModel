#-*- coding: utf-8 -*-

########  本文件实现自定义回归模型的训练，包括
# Part1、无约束条件的模型训练
# Part2、带约束条件的模型训练
######################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
import scipy.optimize as spo

from common import displayRegressionMetrics

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
def yhat(params, X):
    a,b,c,d = params

    ypred = a*X['办公费用'] + b*X['营销费用']**2 + c*X['营销费用'] + d
    return ypred

# 2）自定义损失函数
# from sklearn import metrics
def cost(params, *arg):
    yhat, X, y = arg
    pred = yhat(params, X)

    mse = metrics.mean_squared_error(y, pred)
    return mse

# 3）最优化参数
# import scipy.optimize as spo
params = [10, 20, 3, 4]   #初始值
optResult = spo.minimize(cost, params, args=(yhat, X, y),
            method='trust-constr' )

# print(mthd, optResult)
if optResult['success']:
    bestParams = optResult['x']
    print('\n最优参数：\n',bestParams)
else:
    print('训练失败！method可能不合适！')

# 4.评估模型
y_pred = yhat(bestParams, X)

# from common import displayRegressionMetrics
displayRegressionMetrics(y, y_pred)


# 5.最优算法选取method（假定你不知道的话）
methods = ['Nelder-Mead','CG','BFGS','Newton-CG','dogleg',
            'trust-ncg','trust-crylov','trust-exact',
            'COBYLA','SLSQP','trust-constr',         #有约束
            'Powell','L-BFGS-B','TNC']   #有边界和约束

dictParams = {}     #保留各算法的r2和最佳参数
for mthd in methods:
    try:
        optResult = spo.minimize(cost, params, args=(yhat, X, y),
                    method=mthd )

        if optResult['success']:
            bestParams = optResult['x']
            y_pred = yhat(bestParams, X)
            r2 = metrics.r2_score(y, y_pred)
            dictParams[mthd] = (r2, bestParams)

            # print('\nmethod={},R2={:.4f}'.format(mthd, r2))
        else:
            print('训练失败！method={}'.format(mthd))
    except:
        print('训练失败except！method={}'.format(mthd))

bestScore = -2
bestKey = None
for key in dictParams.keys():
    r2, params = dictParams[key]
    if r2 > bestScore:
        bestScore = r2
        bestKey = key
if bestKey is not None:
    print('最优方法=', bestKey)
    print('模型参数=', dictParams[bestKey])


######################################################################
########  Part2、自定义模型：带约束条件
######################################################################
# 模型：y(t) = base + trend*t + (季节因子)
# 约束条件：季节因子之和=0
# 边界：base(30,50), trend(-5, 5), sv(-100, 100)

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

# 参数初始值
seasons = [1,2,3,4,5,6,7,8,9,10,11,12]
params = [30, 1]
params.extend(seasons)

# 约束条件
# type='eq',表示fun值=0；type='ineq',表示fun值为非负数
cons = ({'type':'eq', 'fun': lambda params: params[2:].sum()})

# 边界，以元组的方式
sv = [(-100,100)]*12        #季节因子的取值范围
bnds = [(30,50),(-5,5)]     #base,trend的取值范围
bnds.extend(sv)

# import scipy.optimize as spo
optResult = spo.minimize(cost, params, 
            args=(yhat2, X, y), 
            method='trust-constr',
            constraints=cons,
            bounds=bnds
            )
# print(optResult)

if optResult['success']:
    bestParams = optResult['x']

    monthCols = [f'{v}月' for v in range(1, len(seasons)+1)]
    sr = pd.Series(
            data=bestParams,
            index=['base','trend'] + monthCols
    )
    print(sr)
else:
    print('训练失败')

# 4.评估模型
y_pred = yhat2(bestParams, X)
displayRegressionMetrics(y, y_pred)

plt.plot(df.index, df['里程（万）'], label='里程(万)')
plt.plot(df.index, y_pred, label='预测值')
plt.legend()
plt.show()

# 5.最优算法（同上，略）



# 算法参数说明
# method指定算法参数
    # method : str or callable, optional Type of solver. 
    # 如果默认，则基于是否有约束条件或边界，选择BFGS,L-BFGS-B, SLSQP三种之一
    # 无约束条件时**Non-Constrained minimization**
        # Nelder-Mead
        # CG    采用非线性共轭梯度算法，一阶导数？
        # BFGS  拟牛顿法，只使用first derivatives，对于非平滑优化有着良好的性能。方法返回近似的海森矩阵，保存在OptimizeResult.hess_inv对象中
        # Newton-CG 缩略牛顿法，使用CG方法来计算搜索方向，适合大规模的问题
        # dogleg    使用dog-leg trust-region算法，需要梯度gradient和海森矩阵Hession
        # trust-ncg 使用牛顿共轭梯度依赖域算法，需要gradient和Hession，适合大规模的问题
        # trust-krylov  使用牛顿GLTR依赖域算法，需要gradient和Hession，适合大规模的问题。
            # 在不确定问题上，比trust-ncg有更少的迭代，中等和大样本量时推荐使用。
        # trust-exact   依赖域算法，近似求解二次方问题，需要gradient和Hession。中小规模问题推荐，有更少的迭代次数
        # 
    # 有边界和约束条件时**Bound-Constrained minimization**
        # Powell    带共轭方向的Powell算法
        # L-BFGS-B  带约束条件的拟牛顿法
        # TNC       也叫牛顿共轭梯度，使用梯度信息，
    # 有约束条件 **Constrained Minimization**
        # COBYLA    线性近似法.约束fun可返回单个值，或数组，或列表
        # SLSQP     序列的最小二乘法
        # trust-constr  依赖域算法，适合大规模问题。
    # **Finite-Difference Options**
        # trust-constr

    # **Custom minimizers**
        # 自定义算法
        
    
    # bounds参数：用于L-BFGS-B, TNC, SLSQP, Powell, trust-constr方法
    # bounds : sequence or Bounds
    # 两种方式指定边界
        #1. Bounds类实例
        #2. 每一个x的(min, max)的序列对. 
        #3. None表示没有边界

    # constraints约束条件,限于COBYLA, SLSQP, trust-constr算法
    # constraints : {Constraint, dict} or List of {Constraint, dict}, 

    # trust-constr算法的约束可以是单个对象或对象列表
    # 约束条件有:
        # `LinearConstraint`    线性约束
        # `NonlinearConstraint` 非线性约束

    # COBYLA, SLSQP算法的约束是字典列表，字典包含key值
        #     type : str，取'eq'表示fun值等于0,'ineq'表示fun值大于或等于0（非负数）
        #     fun : callable，约束的函数定义
        #     jac : callable, optional，The Jacobian of `fun` (only for SLSQP).
        #     args : sequence, optional，传给fun和jacobian的其它参数
    # 注意：COBYLA只支持'ineq'约束

