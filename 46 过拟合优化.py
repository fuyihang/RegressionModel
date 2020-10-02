#-*- coding: utf-8 -*-

########  本文件实现过拟合回归模型优化，包括
# Part1、过拟合评估：学习曲线
# Part2、Ridge岭回归
# Part3、Lasso套索回归
# Part4、ElasticNet弹性网络回归
# Part5、超参优化
    # 1）手工遍历
    # 2）交叉验证类RidgeCV/LassoCV/ElasticNetCV
    # 3）网格搜索类GridSearchCV
    # 4）随机搜索类RandomizedSearchCV
######################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from common import displayRegressionMetrics

######################################################################
########  Part1、过拟合检验
######################################################################
# 过拟合：在训练集上的质量较高，但在测试集上的质量偏低
# 用户学习曲线来判断过拟合

# 准备数据
filename = '回归分析.xlsx'
sheet = '销售额'
df = pd.read_excel(filename, sheet)

intCols = ['办公费用', '营销费用']
target = '销售额'

X = df[intCols]
y = df[target]

from sklearn.linear_model import LinearRegression

mdl = LinearRegression()
mdl.fit(X, y)


from common import plot_learning_curve

plot_learning_curve(mdl, X, y)

######################################################################
########  
######################################################################
filename = '回归分析.xlsx'
sheet = '身高年龄与体重'
df = pd.read_excel(filename, sheet_name=sheet)

cols = ['身高', '年龄']
target = '体重'

# 2、数据处理
X = df[cols]
y = df[target]

from sklearn.preprocessing import PolynomialFeatures

n = 4    #表示最高n次方
po = PolynomialFeatures(degree=n, include_bias=False)
fit_X = po.fit_transform(X)

# 构造列标题：cols
cols = ['x%s'% x for x in range(1, fit_X.shape[1]+1)]

# 3、训练模型
from sklearn.linear_model import LinearRegression

mdl = LinearRegression()
mdl.fit(fit_X, y)

sr = pd.Series(name='回归系数',
        data=[mdl.intercept_]+mdl.coef_.tolist(),
        index=['常数']+cols)
print(sr)

# 4、评估模型
y_pred = mdl.predict(fit_X)
displayRegressionMetrics(y, y_pred, X.shape)

# 上述R2过高，查看相关系数矩阵，发现存在共线性
print(np.corrcoef(fit_X))
plot_learning_curve(mdl, fit_X, y)


######################################################################
########  Part2、Ridge岭回归
######################################################################
# 带L2正则项
# 目标函数：||y - Xw||^2_2 + alpha * ||w||^2_2

from sklearn.linear_model import Ridge 

alpha = 10.0
mdl = Ridge(alpha=alpha) #cholesky
mdl.fit(fit_X, y)

print('score=', mdl.score(fit_X, y))

sr = pd.Series(name='回归系数',
        data=[mdl.intercept_]+mdl.coef_.tolist(),
        index=['常数']+cols)
print('Ridge回归系数：\n',sr)

# 评估
y_pred = mdl.predict(fit_X)
displayRegressionMetrics(y_pred, y, fit_X.shape)
# R2变小，系数也相对小
# alpha越大，R2越小，系数也在收缩越小

# 超参优化（参考后面部分）


# 相关类：
    # sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True, normalize=False, 
    #       copy_X=True, max_iter=None, tol=0.001, solver=’auto’, random_state=None)
    # 参数说明：
    # alpha=1.0 正则化强度，值越大，表示收缩量越大，对共线性的处理会更好 
    # fit_intercept=True, 是否计算截距，即保存回归常数
    # normalize=False, 是否要归一化
    # copy_X=True, 是否将复制X，否则可能被覆盖。
    # max_iter=None, 迭代次数。
    #   对于‘sparse_cg'和'lsqr'，默认值由scipy.sparse.linalg确定；
    #   对于'sag'求解器，默认值为1000.
    # tol=0.001, 解的精度
    # solver：{‘auto’，’svd’，’cholesky’，’lsqr’，’sparse_cg’，’sag’} 
    # 用于计算的求解方法： 
    #   ‘auto’根据数据类型自动选择求解器。 
    #   ‘svd’使用X的奇异值分解来计算Ridge系数。对于奇异矩阵比’cholesky’更稳定。 
    #   ‘cholesky’使用标准的scipy.linalg.solve函数来获得闭合形式的解。 
    #   ‘sparse_cg’使用在scipy.sparse.linalg.cg中找到的共轭梯度求解器。
    #        作为迭代算法，这个求解器比“cholesky”更合适对大规模数据（设置tol和max_iter的可能性）处理。 
    #   ‘lsqr’使用专用的正则化最小二乘常数scipy.sparse.linalg.lsqr。它是最快的，它还使用迭代过程。 
    #   ‘sag’使用随机平均梯度下降。它也使用迭代过程，并且当n_samples和n_feature都很大时，通常比其他求解器更快。
    #           注意，“sag”快速收敛仅在具有近似相同尺度的特征上被保证。您可以使用sklearn.preprocessing的缩放器预处理数据。 
    #           ’saga'为‘sag'的优化。
    #   所有最后5个求解器支持密集和稀疏数据。但是，当fit_intercept为True时，只有’sag’和'sparse_cg'支持稀疏输入。 

    # random_state=None。 伪随机数生成器的种子，仅用于'sag'求解器
    # )

######################################################################
########  Part3、Lasso套索回归
######################################################################
# 带L1正则项
# 目标函数：(1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

from sklearn.linear_model import Lasso

alpha = 10.0
mdl = Lasso(alpha=alpha)
mdl.fit(fit_X, y)

sr = pd.Series(name='回归系数',
        data=[mdl.intercept_]+mdl.coef_.tolist(),
        index=['常数']+cols)
print('Lasso回归系数：\n', sr)

# 评估
y_pred = mdl.predict(fit_X)
displayRegressionMetrics(y_pred, y, fit_X.shape)
# 回归系数，多数为0

# 超参优化（参考后面部分）

######################################################################
########  Part4、ElasticNet弹性回归
######################################################################
# 同时带L1和L2正则项
# 目标函数：(1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

from sklearn.linear_model import ElasticNet

alpha = 3
l1_ratio = 0.9

mdl = ElasticNet(alpha=alpha, l1_ratio= l1_ratio)
mdl.fit(fit_X, y)

sr = pd.Series(index=['常数']+cols,
        data=[mdl.intercept_]+mdl.coef_.tolist())
print('ElasticNet回归系数：\n', sr)

# 评估
y_pred = mdl.predict(fit_X)
displayRegressionMetrics(y_pred, y, fit_X.shape)
# 结合前两者的优点

# 超参优化（参考后面部分）


######################################################################
########  Part5、超参优化
######################################################################
# 超参优化四种方式：
# 1）手工遍历
# 2）交叉验证类RidgeCV/LassoCV/ElasticNetCV
# 3）网格搜索类GridSearchCV
# 4）随机搜索类RandomizedSearchCV
# ...

X = fit_X

# 构造列标题：cols
cols = ['x%s'% x for x in range(1, X.shape[1]+1)]

# 先利用训练集进行
mdl = Ridge()      #默认alpha=1.0
mdl.fit(fit_X, y)
y_pred = mdl.predict(fit_X)
displayRegressionMetrics(y, y_pred, fit_X.shape)

############################
######## 1)超参优化：手工遍历
############################

from sklearn.model_selection import cross_val_score

scores = []
Params = []

mdl = Ridge()
alphas = np.linspace(0.01, 1, 100)     #0.01~10之间取100个值
for a in alphas:
    mdl.set_params(alpha=a)
    r2s = cross_val_score(mdl, X, y, cv=5, scoring='r2')
    # print(r2s)
    scores.append(r2s.mean()) 
    Params.append(a)
# print(scores)

# 选出最优超参
idx = np.argmax(scores)
print('最优得分：', scores[idx])
print('最优参数：', Params[idx])

############################
######## 2)超参优化：交叉验证类
############################
# 如RidgeCV

from sklearn.linear_model import RidgeCV

alphas = np.linspace(0.1, 10, 100)
# alphas = [0.01, 0.1, 2.0, 5.0]
mdl = RidgeCV(alphas = alphas, cv=5)
mdl.fit(X, y)

# 返回最优参数及最优最优模型
print('最优alpha=', mdl.alpha_)
print('score = ', mdl.score(X, y))

sr = pd.Series(name='RidgeCV系数',
                data = [mdl.intercept_] + mdl.coef_.tolist(),
                index = ['常数']+cols)
print(sr)

pred = mdl.predict(X)
displayRegressionMetrics(y, pred, X.shape)

############################
######## 弹性网络 超参优化 ElasticNetCV
############################

print('\n=======ElasticNetCV==========')
from sklearn.linear_model import ElasticNetCV

alphas = np.linspace(1, 10, 100)
ratios = np.linspace(0, 1.0, 10)

mdl = ElasticNetCV(alphas=alphas, l1_ratio=ratios)
mdl.fit(X, y)

#返回最优参数,此时mdl即为最优模型
print('最优alpha=', mdl.alpha_)
print('最优l1_ratio=', mdl.l1_ratio_)
print('score = ', mdl.score(X, y))

sr = pd.Series(name='ElasticNetCV系数',
                data = [mdl.intercept_] + mdl.coef_.tolist(),
                index = ['常数']+cols)
print(sr)

y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)


############################
######## 3)超参优化：网格搜索 GridSearchCV
############################
# 穷尽搜索最优超参

from sklearn.model_selection import GridSearchCV

# 构建参数字典
params = {'alpha':np.linspace(1, 100, 50),
            'l1_ratio':np.linspace(0.01,1,50) }

mdl = ElasticNet()
grid = GridSearchCV(estimator = mdl, 
            param_grid = params, cv=5, scoring='r2')
grid.fit(X, y)

print('best_score_:',grid.best_score_)  # 获取最佳度量值
print('best_params_：',grid.best_params_)  # 最佳参数字典

mdl = grid.best_estimator_      #返回最优模型
# 保存最优模型

y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)

# 其它属性
# print('网格搜索-度量记录：',grid.cv_results_)  # 包含每次训练的相关信息
# print(sorted(grid.cv_results_.keys()))
# print('best_index_:', grid.best_index_)  #最佳索引
# print('n_splites_:', grid.n_splits_)    #交叉验证的分割数
# print('refit_time:', grid.refit_time_)  #重训练所用时间（秒数）

# 相关函数
    # sklearn.model_selection.GridSearchCV(estimator, param_grid, scoring=None, n_jobs=None, 
    #       iid=’warn’, refit=True, cv=’warn’, verbose=0, pre_dispatch=‘2*n_jobs’, 
    #       error_score=’raise-deprecating’, return_train_score=False)
    # estimator : estimator object.模型对象
    # param_grid : dict or list of dictionaries
    #   字典，字典列表
    # scoring : string, callable, list/tuple, dict or None, default: None
    #   评估指标（可单个，也可多个）。
    #   默认，调用模型的score方法的返回值
    # n_jobs : int or None, optional (default=None)
    #   并行运算的数量。None表示一般表示为1或者使用joblib.parallel_backend上下文
    #   -1表示可使用所有的处理器，实现并行计算。
    # pre_dispatch : int, or string, optional
    #   并行执行时任务数，减少数量可避免内存消耗.
    #   None：任务即时创建和启动（spawned）. 
    #   int：启动的准确的数量
    #   string：表达式如‘2*n_jobs’
    # iid : boolean, default=’warn’
    #   返回每折的平均得分？
    # cv : int, cross-validation generator or an iterable, optional
    #   分割策略:
    #   None：默认3折交叉
    #   integer：指定折数
    # CV splitter,
    # refit : boolean, string, or callable, default=True
    #   重拟合最优参数的模型，并返回：
    #   best_estimator_ 最优模型
    #   best_index_
    #   best_score_
    #   best_params_
    # verbose : integer
    #   控制冗余（verbosity），越高信息越多
    # error_score : ‘raise’ or numeric
    #   当调用score函数时，发生错误时的处理。
    #   或者抛出异常，或者返回指定的值，或者返回np.nan
    # return_train_score : boolean, default=False
    #   False：则属性cv_results_不会包含训练集得分


############################
######## 超参优化：随机搜索 RandomizedSearchCV
############################
# 穷尽搜索最优超参

from sklearn.model_selection import RandomizedSearchCV

# 构建参数字典
params = {'alpha':[1e-3, 1e-2, 1e-1, 1, 10, 100],
            'l1_ratio':[0.1, 1] }

mdl = ElasticNet()
grid = RandomizedSearchCV(estimator = mdl, 
            param_distributions = params, cv=5, scoring='r2')
grid.fit(X, y)

print('best_score_:',grid.best_score_)  # 获取最佳度量值
print('best_params_：',grid.best_params_)  # 最佳参数字典

mdl = grid.best_estimator_      #返回最优模型
# 保存最优模型

y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)

# 相关函数
    # sklearn.model_selection.RandomizedSearchCV(estimator, param_distributions, n_iter=10, 
    #       scoring=None, n_jobs=None, iid=’warn’, refit=True, cv=’warn’, verbose=0, 
    #       pre_dispatch=‘2*n_jobs’, random_state=None, error_score=’raise-deprecating’, 
    #       return_train_score=False)
    # estimator : estimator object.
    # param_distributions : dict。参数字典
    # n_iter : int, default=10。抽样次数
    # scoring : string, callable, list/tuple, dict or None, default: None
    # n_jobs : int or None, optional (default=None)
    # pre_dispatch : int, or string, optional
    # iid : boolean, default=’warn’
    # cv : int, cross-validation generator or an iterable, optional
    # CV splitter,
    # refit : boolean, string, or callable, default=True
    # verbose : integer
    # random_state : int, RandomState instance or None, optional, default=None
    # error_score : ‘raise’ or numeric
    # return_train_score : boolean, default=False

