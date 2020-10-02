#-*- coding: utf-8 -*-
# conda install scikit-learn

########  本文件实现其它回归模型，包括
# Part1、线性回归：statsmodels.api
# Part2、模型自动筛选自变量

# Part4、决策回归树(CART回归树)
# Part5、神经网络回归(ANN-MLP)
# Part6、支持向量回归(SVR)

######################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from common import displayRegressionMetrics
from common import featureSelection


######################################################################
########  Part1、线性回归 使用statsmodels.api
######################################################################

# 1、读取数据
filename = 'Telephone.csv'
df = pd.read_csv(filename, encoding='gbk')
# print(df.columns.tolist())

# 2、特征工程
intCols = ['年龄','收入', '家庭人数', '开通月数','电子支付']
catCols = ['居住地', '婚姻状况', '教育水平', '性别']
cols = intCols + catCols

target = '消费金额'
y = df[target]

# 3、训练模型
import statsmodels.formula.api as smf

formula = '{} ~ {}'.format(target, '+'.join(cols))

module = smf.ols(formula, df)
results = module.fit()
# print(results.summary())    #回归结果

sr = results.params
print(sr)       # 回归系数,返回Series

# 4、评估模型

# 1）方程显著性检验：表示是否可以使用回归分析
if results.f_pvalue < 0.05:     #方程显著性
    print('方程具有显著性，模型可用.')

# 2）回归系数的显著性检验：自变量与因变量是否有显著影响
# print(results.tvalues)
sr = results.pvalues    #查看所有回归系数的显著性，Series
# print(sr)

sr.drop('Intercept', inplace=True)
cond = sr < 0.05
cols = sr[cond].index.tolist()
print('显著影响因子：', cols )

# 3）模型评估指标（拟合程度）
# 拟合程度（越接近于1表示模型越好）
print('R2={:.4}, adjR2={:.4}'.format(results.rsquared, results.rsquared_adj))

# 信息准则指标(带惩罚项，越小表示模型越好)
print('AIC={}, BIC={}'.format(results.aic, results.bic))    #信息准则指标

y_pred = results.fittedvalues
displayRegressionMetrics(y, y_pred)

# 4) 共线性检验：
# results.condition_number值过大，有可能存在共线性

# 5）残差检验
# results.resid     #残差 = y - y_pred

# 5、模型优化

# 6、应用模型
# 1）预测历史数据
y_pred = results.fittedvalues

# 2）预测新数据
XX = df.loc[10:11,:]
pred = results.predict(XX)
print(pred)

# 3）保存模型
fname = 'out.mdl'
results.save(fname, remove_data=True)
# remove_data表示移除预测数值fittedvalues等，默认为False要保存
# 我认为没有必要，所以此处设置为True

# 4)加载模型
from statsmodels.regression.linear_model import RegressionResults

fname = 'out.mdl'
results = RegressionResults.load(fname)

# 5）预测新数据(略)
print(results.params)



######################################################################
########  Part2、模型自动筛选自变量
######################################################################
# 当加入变量时，模型质量变化不大，则说明此变量不用进入模型

cols = intCols + catCols

import itertools
AICs = {}
for k in range(len(cols)):
    # combinations()函数返回一个元组，列表是由x的k个字段组成的集合
    for var in itertools.combinations(cols, k+1):
        varCols = list(var)

        formula = '{} ~ {}'.format(target,
                    '+'.join(varCols) )
        # print(formula, varCols)
        mdl = smf.ols(formula, df[[target]+varCols])
        results = mdl.fit()
        AICs[var] = results.aic
print(AICs)

#找出字典中最小AIC值的Key，即字段列表
cols = min(AICs, key=AICs.get)
cols = list(cols)

# 然后，再利用这些字段建模
formula = '{} ~ {}'.format(target, '+'.join(cols))

mdl = smf.ols(formula, df)
results = mdl.fit()
print(results.params)       #回归系数，包括常量

# 其他略

######################################################################
########  Part4、决策回归树(CART回归树)
######################################################################
# 决策树用于回归问题

# 1、读取数据
filename = 'Telephone.csv'
df = pd.read_csv(filename, encoding='gbk')
print(df.columns.tolist())

# 2、特征工具
# 1）特征标识/筛选
intCols = ['年龄','收入', '家庭人数', '开通月数']
catCols = ['居住地', '婚姻状况', '教育水平', '性别']

target = '消费金额'
y = df[target]

# 2）类别变量数字化
from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder(dtype='int')
X_cats = enc.fit_transform(df[catCols])

# 3）合并
X = np.concatenate([X_cats, df[intCols]], axis=1)
cols = catCols + intCols

# 3、建模
from sklearn.tree import DecisionTreeRegressor

mdl = DecisionTreeRegressor(random_state = 10)
mdl.fit(X, y)

# 显示特征重要性
sr = pd.Series(mdl.feature_importances_, index = cols, name='决策树')
sr.sort_values(ascending=False, inplace=True)
sr.plot(kind='bar', title='特征重要性')
# plt.show()

# 4、评估
y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)

# 5、超参优化
from sklearn.model_selection import GridSearchCV

params = {
            'criterion':['mse','friedman_mse','mae'],
            'max_depth': range(3,14,1), 
            'min_samples_split':range(20,210,20), 
            'min_samples_leaf':range(10,60,10)
        }
mdl = DecisionTreeRegressor(splitter='best',
        random_state=10)
grid = GridSearchCV(estimator=mdl, param_grid=params, cv=5)
grid.fit(X, y)

print("最优模型得分：",grid.best_score_)
print("最优超参：",grid.best_params_)

# 可保存最优超参，及最优模型，以便后续使用
bestParams = grid.best_params_
mdl = grid.best_estimator_

y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)

# 6、应用模型（略）
# 保存决策树
# 决策树解决
# 保存模型


# DecisionTreeClassifier(
    # ccp_alpha=0.0, class_weight=None, criterion='gini',
    # max_depth=None, max_features=None, max_leaf_nodes=None,
    # min_impurity_decrease=0.0, min_impurity_split=None,
    # min_samples_leaf=1, min_samples_split=2,
    # min_weight_fraction_leaf=0.0, presort='deprecated',
    # random_state=None, splitter='best')
# DecisionTreeRegressor
    # 回归树与分类树最大的判别在于criterion的取值
    # criterion : {"mse", "friedman_mse", "mae"} 最优特征选择和最佳分割点的判断指标
        # mse:误差平方和
        # friedman_mse:" 优化过的mse
        # mae:平均绝对值误差

    # 重要参数
        # criterion：gini基尼系数或者entropy信息熵，
        #           前者即CART算法，后者即类似ID3, C4.5的最优特征选择方法。
        # splitter： 
        #       best：寻找最佳分割点，适合于样本量不大的场景
        #       random 随机寻找分割点，超大样本量时
        # random_state=None, 这个一般取固定值，比如10.
        # presort='deprecated',丢弃。表示在拟合之前，是否预分数据来加快树的构建。

    # 预剪枝参数
        # max_features：选取的特征数量(我建议手工选取)
        #       None（所有），默认
        #       log2，sqrt，N 按此计算特征数量
        #       特征小于50的时候一般使用所有的
        # max_depth：  int or None, optional (default=None) 
        #       树的最大深度。适当的设置，可以避免过拟合。常用的可以取值10-100之间。
        # min_samples_split：节点的最小样本量
        #       如果某节点的样本数少于min_samples_split，则不会继续再尝试选择最优特征来进行划分
        #       如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。
        # min_samples_leaf： 子节点的最小样本量。
        #       限制叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝
        # max_leaf_nodes： 最大叶子节点数，可以防止过拟合，
        #       默认是"None”，即不限制最大的叶子节点数。
        #       如果加了限制，算法会建立在最大叶子节点数内最优的决策树。
        #       但是如果特征变量多的话，可以加以限制，可以防止过拟合。
        # min_impurity_decrease=0.0, 节点划分最小不纯度。[float]
        #       限制决策树的增长，如果划分后不纯度的变化小于这个值，就不再生长。 
        # min_impurity_split=None,  丢弃了。

    # 后剪枝参数
        # ccp_alpha=0.0, 非负的float。最小剪枝系数，默认为0
        #        该参数用来限制树过拟合的剪枝参数。
        #       ccp_alpha=0时，决策树不剪枝；
        #       ccp_alpha越大，越多的节点被剪枝。

    # 样本权重
        # class_weight： 指定各类别的的权重，解决正负样本不平衡的问题
        #       防止训练集某些类别的样本过多导致训练的决策树过于偏向这些类别。
        #       如果使用“balanced”，则算法会自己计算权重，样本量少的类别所对应的样本权重会高。
        # min_weight_fraction_leaf：子节点所有样本权重和的最小值，解决缺失过大的问题
        #       如果小于这个值，则会和兄弟节点一起被剪枝。默认是0，就是不考虑权重问题。
        #       一般有较多样本缺失，或者分类树样本的分布类别偏差很大，就会引入样本权重。

    # 重要属性
        # classes_ : array of shape = [n_classes] or a list of such arrays
        #       类标签的取值范围
        # feature_importances_ : array of shape = [n_features]
        #       返回特征的重要程度
        # max_features_ : int, 最大特征的个数
        # n_classes_ : int or list 类变量取值的个数
        # n_features_ : int 训练前的特征数
        # n_outputs_ : int 训练后输出的数量
        # tree_ : Tree object   返回树对象
        #       可用help(sklearn.tree._tree.Tree)来看到对象的属性


######################################################################
########  Part5、神经网络回归(ANN-MLP)
######################################################################
# 神经网络用于回归问题

# 1、读取数据
filename = 'Telephone.csv'
df = pd.read_csv(filename, encoding='gbk')
print(df.columns.tolist())

# 2、特征工具
# 1）特征标识/筛选
intCols = ['年龄','收入', '家庭人数', '开通月数']
catCols = ['居住地', '婚姻状况', '教育水平', '性别']

target = '消费金额'
y = df[target]

# 2）类别变量-->哑变量
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(dtype='int', sparse=False)
X_cats = enc.fit_transform(df[catCols])
TmpCols = []
for col in enc.categories_:
    TmpCols.extend(col.tolist())
print(TmpCols)

# 3）数值型变量要标准化
from sklearn.preprocessing import StandardScaler

enc = StandardScaler()
X_ints = enc.fit_transform(df[intCols])

# 3）合并
X = np.concatenate([X_cats, X_ints], axis=1)
cols = TmpCols + intCols
# print(X.shape)

# 3、建模
from sklearn.neural_network import MLPRegressor

mdl = MLPRegressor(random_state = 10, 
        hidden_layer_sizes=(50, 25, 10))
mdl.fit(X, y)

# 4、评估
y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)

# 5、超参优化
from sklearn.model_selection import GridSearchCV

params = {
    'activation':['logistic','tanh','relu'],
    'solver':['sgd'],
    'hidden_layer_sizes': [(12, 5), (128,7), (50, 7, 7)],
    # 'epsilon': [1e-3, 1e-7],
    'alpha':[0.1, 0.2, 0.8, 1.0] }
mdl = MLPRegressor()
grid = GridSearchCV(estimator=mdl, param_grid=params, cv=5)
grid.fit(X, y)

# 保存
bestParams = grid.best_params_
mdl = grid.best_estimator_

y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)

# 输出神经网络的详细信息：
print('总共层数：', mdl.n_layers_)
print('输入层：')
print('\t节点数=', mdl.n_features_in_)

print('中间层：', mdl.hidden_layer_sizes)
for i, n in enumerate(mdl.hidden_layer_sizes):
    print('\t中间层{}'.format(i+1))
    print('\t\t节点数:{}'.format(n))
    print('\t\t加法器shape：', mdl.coefs_[i].shape)

print('输出层：')
print('\t节点数=', mdl.n_outputs_)
print('\t激活函数=', mdl.out_activation_)


# sklearn.neural_network.MLPRegressor(
        # hidden_layer_sizes=(100, ), 
        # activation=‘relu’, solver=‘adam’, 
        # alpha=0.0001,                 #正则化系数
        # batch_size=‘auto’, 
        # learning_rate=‘constant’, learning_rate_init=0.001, 
        # power_t=0.5, max_iter=200, shuffle=True, random_state=None, 
        # tol=0.0001, verbose=False, warm_start=False, momentum=0.9, 
        # nesterovs_momentum=True, early_stopping=False, 
        # validation_fraction=0.1, 
        # beta_1=0.9, beta_2=0.999, epsilon=1e-08)


######################################################################
########  Part6、支持向量回归(SVR)
######################################################################
# 支持向量用于回归问题

# 数据集处理同上（略）

# 3、训练模型
from sklearn.svm import SVR

mdl = SVR(kernel='rbf', C=1.0)
mdl.fit(X, y)

# 4、评估模型
y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)

# 5、超参优化
from sklearn.model_selection import GridSearchCV

params = [{'kernel':['rbf','sigmoid'],
            # 'gamma':np.linspace(0, 0.001, 50),
            'C':np.linspace(0.1, 5, 20)
        },
        {'kernel':['poly'],
            'degree':range(2, 8),
            # 'gamma':np.linspace(0, 0.001, 50),
            'C':np.linspace(0.1, 5, 20)
        }]

grid = GridSearchCV(mdl, param_grid=params, cv=5)
grid.fit(X, y)

bestParams = grid.best_params_
mdl = grid.best_estimator_

# 预测
y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)

# 模型信息
vts = mdl.support_vectors_  # 获取支持向量列表
idxs = mdl.support_         # 支持向量的索引列表
vtsNum = mdl.n_support_     # 支持向量的个数


# sklearn.svm.SVR（kernel ='rbf'，degree = 3，
        # gamma ='auto_deprecated'，coef0 = 0.0，tol = 0.001，
        # C = 1.0，epsilon = 0.1，shrinking = True，cache_size = 200，
        # verbose = False，max_iter = -1 ）

    # kernel ： string，optional（default ='rbf'）
        # 指定要在算法中使用的内核类型。
        # 它必须是'linear'，'poly'，'rbf'，'sigmoid'，'precomputed'或者callable之一。
    # degree： int，可选（默认= 3）
        # 多项式核函数的次数（'poly'）。被所有其他内核忽略。
    # gamma ： float，optional（默认='auto'）
        # 'rbf'，'poly'和'sigmoid'的核系数。
        # 当前默认值为'auto'，它使用1 / n_features，
        # 如果gamma='scale'传递，则使用1 /（n_features * X.std（））作为gamma的值。

    # coef0 ： float，optional（默认值= 0.0）
        # 核函数中的独立项。它只在'poly'和'sigmoid'中很重要。
    # tol ： float，optional（默认值= 1e-3）
        # 容忍停止标准。
    # C ： float，可选（默认= 1.0）
        # 惩罚参数C.
    # epsilon ： float，optional（默认值= 0.1）
        # Epsilon在epsilon-SVR模型中。它指定了epsilon-tube，其中训练损失函数中没有惩罚与在实际值的距离epsilon内预测的点。
    # shrinking = True
        # 收缩 ： 布尔值，可选（默认= True）
        # 是否使用收缩启发式。
    # cache_size ： float，可选
        # 指定内核缓存的大小（以MB为单位）。
        # 详细说明 ： bool，默认值：False
        # 启用详细输出。请注意，此设置利用libsvm中的每进程运行时设置，如果启用，则可能无法在多线程上下文中正常运行。
    # max_iter ： int，optional（默认值= -1）
        # 求解器内迭代的硬限制，或无限制的-1
