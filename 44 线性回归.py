#-*- coding: utf-8 -*-
# conda install scikit-learn

########  本文件实现回归预测模型，包括
# Part1、线性回归（一元/多元）
# Part2、线性回归（带类别变量的线性回归）
# Part3、模型评估方法
# Part4、过拟合检验
# Part5、欠拟合解决方法：多项式回归
# Part6、过拟合解决方法：正则化（Ridge, Lasso, ElasticNet）
# Part7、超参优化（RidgeCV, LassoCV, ElasticNetCV）
# Part8、实战练习：线性回归
######################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.linear_model import RidgeCV,LassoCV,ElasticNetCV

from sklearn.model_selection import train_test_split,learning_curve

from sklearn import metrics
def displayRegressionMetrics(y_true, y_pred, adjVal=None):
    '''
    \n功能：计算回归的各种评估指标。
    \n参数：y_true:真实值
         y_pred:预测值
         adjVal:输入的shape参数(n,p)，其中n是样本量，p是特征数
            默认None表示是一元回归；
    \n返回：各种指标，字典形式
    '''
    # 评估指标：R^2/adjR^2, MAPE, MAE，RMSE
    mts = {}
    #一元回归，计算R^2；
    mts['R2'] = metrics.r2_score(y_true, y_pred)
    # 多元回归，计算调整R^2
    if (adjVal != None) and (adjVal[1] > 1):
        n, p = adjVal
        mts['adjR2']  = 1-(1-mts['R2'])*(n-1)/(n-p-1)

    mts['MAPE'] = (abs((y_pred-y_true)/y_true)).mean()
    mts['MAE'] = metrics.mean_absolute_error(y_true, y_pred)
    MSE = metrics.mean_squared_error(y_true, y_pred)
    mts['RMSE'] = np.sqrt(MSE)
    
    # 格式化，保留小数点后4位
    for k,v in mts.items():
        mts[k] = np.round(v, 4)
    
    # 特别处理,注意变成了字符串
    mts['MAPE'] = '{0:.2%}'.format(mts['MAPE']) 

    print('回归模型评估指标：\n', mts)
    # 也可以手工计算
    # ssr = ((y_pred-y_true.mean())**2).sum()
    # sst = ((y_true-y_true.mean())**2).sum()
    # mts['R2'] = ssr/sst
    # mts['adjR2'] = 1- ((sst-ssr)/(n-p-1))/(sst/(n-1))
    # mts['MAE'] = (abs(y_pred-y_true)).mean()
    # mts['RMSE'] = np.sqrt(((y_pred-y_true)**2).mean())

    # # 残差检验：均值为0，正态分布，随机无自相关
    # resid = y_true - y_pred         #残差
    # z,p = stats.normaltest(resid)   #正态检验
    
    return


######################################################################
########  Part1、线性回归
######################################################################

############################
######## 一元线性回归 ########
############################

# 1、准备数据集
import pandas as pd

filename = '回归分析.xlsx'
sheet = '销售额'
df = pd.read_excel(filename, sheet)
print(df.columns.tolist())

# 2、数据预处理

# 1）数据选取
X = df[['营销费用']]
y = df['销售额']

# 2）可视化观察
import matplotlib.pyplot as plt

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
print('回归常数：', mdl.intercept_)
print('回归系数：', mdl.coef_.tolist())

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
# 1）预测
# 此处X可以是任何待预测的样本集
y_pred = mdl.predict(X)

# 2）可视化观察
plt.scatter(X, y)               # 画原始数据散点图
plt.plot(X, y_pred, color='k')  #画回归直线

plt.title('营销费用vs销售额')
plt.xlabel('营销费用')
plt.ylabel('销售额')
plt.show()

# 3）保存模型
import joblib
filename = 'out.mdl'
joblib.dump(mdl, filename)

# 4）加载模型
mdl = joblib.load(filename)

# 5）预测-新值
XX = [[8000],[9000]]
pred = mdl.predict(XX)
print(pred)

############################
######## 多元线性回归 ########
############################
# 建立办公费用、营销费用与销售额的多元回归模型

# 1、读取数据（同上，略）
# filename = '回归分析.xlsx'
# sheet = '销售额'
# df = pd.read_excel(filename, sheet)

# 2、数据预处理
cols = ['办公费用', '营销费用']
target = '销售额'

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
# 6、应用模型（略）


########练习：建立身高、年龄与体重的多元回归模型#######

filename = '回归分析.xlsx'
sheet = '身高年龄与体重'
df = pd.read_excel(filename, sheet)
print(df.columns.tolist())


######################################################################
########  Part2、带类别自变量的线性回归
######################################################################
# 如果自变量为类别变量，需要转化成哑变量
# 转换原则：变量取k个值，使用k-1个哑变量

# 1、读入数据
filename = '回归分析.xlsx'
sheet = '汽车季度销量'
df = pd.read_excel(filename, sheet_name=sheet)
print(df.columns.tolist())

# 2、数据处理
# 1)选择属性
catCols = ['季度']
intCols = ['GNP','失业率','利率']
target = '销量'

# 2）如果要预测，则需要错位/移位，并去除首行
shiftCols = ['GNP','失业率','利率']
df[shiftCols] = df[shiftCols].shift(periods = 1, axis=0)
df.drop(0,axis=0,inplace=True)

# 3）哑变量转换
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(dtype='int', drop='first',sparse=False)
X_ = enc.fit_transform(df[catCols])
print(X_.shape)

# enc.categories_属性中，只有一个数组
dfCats = pd.DataFrame(X_, 
                index=df.index, 
                columns=enc.categories_[0][1:])
print(dfCats.head())

# 4）合并连续变量和哑变量
dfCols = pd.concat([dfCats, df[intCols] ], axis=1)
print(dfCols.head())

X = dfCols
y = df[target]

# 3、训练模型
mdl = LinearRegression()
mdl.fit(X, y)

sr = pd.Series(name='回归系数',
            data=[mdl.intercept_] + mdl.coef_.tolist(),
            index=['常数']+ dfCols.columns.tolist())
print(sr)

# 4、评估模型（略）
print('score=', mdl.score(X, y))
y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)

# 5、模型优化（略）

# 6、模型应用
# 预测87年第一季节的汽车销量
data = [['第一季度', 4716, 6.9, 7.4]]
dfx = pd.DataFrame(data=data,columns=catCols+intCols)
print(dfx)

# 哑变量化，并合并
x_ = enc.transform(dfx[catCols])

x = np.concatenate([x_, dfx[intCols]],axis=1)
pred = mdl.predict(x)
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
# 2）类别型变量进行哑变量化
enc = OneHotEncoder(dtype='int', drop='first', sparse=False)
X_ = enc.fit_transform(df[catCols])
print(X_.shape)

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

grouped = df.groupby('性别')
sr = grouped[target].mean()
plt.bar(sr.index, sr.values)
plt.show()

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


######################################################################
########  Part3、模型评估方法
######################################################################

filename = '回归分析.xlsx'
sheet = '销售额'
df = pd.read_excel(filename, sheet)
print(df.columns.tolist())

X = df[['办公费用','营销费用']]
y = df['销售额']

############################
######## 留出法Houd-out，也称简单交叉验证，即1折交叉验证
############################
# 训练集 vs 测试集

print('\n=======模型评估：训练集vs测试集===========')
from sklearn.model_selection import train_test_split

# test_size为0~1的小数时，表示百分比，默认为0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

mdl = LinearRegression()
mdl.fit(X_train, y_train)

print('训练集r2:=', mdl.score(X_train, y_train))
print('测试集r2:=', mdl.score(X_test, y_test))

############################
######## 交叉验证（cross_val_score）
############################
# 进行k次验证

print('\n======模型评估：K折交叉验证============')
from sklearn.model_selection import cross_val_score

mdl = LinearRegression()

# scoring的取值请参考：sklearn.metrics.SCORERS.keys()
scoring = 'r2'    #'r2'
scores = cross_val_score(mdl, X, y, cv=5, scoring=scoring)
miu = scores.mean()
delt =  scores.std()
print('\n平均值:{0:.2f},标准差:{1:.2f}'.format(miu, delt))
print('置信区间%95：[{:.2f},{:.2f}]'.format(miu-2*delt, miu+2*delt))


############################
######## 交叉验证（手工）
############################
# 下面的评估结果与上面cross_val_score是一样的

print('\n=======模型评估：手工 K折交叉验证=============')
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

######################################################################
########  Part4、过拟合检验
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

mdl = LinearRegression()
mdl.fit(X, y)

from sklearn.model_selection import learning_curve

# 画学习曲线
def plot_learning_curve(estimator, X, y, cv=None, scoring = 'r2'):
    train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv = cv,scoring=scoring, 
            train_sizes=np.linspace(0.1, 1.0, 50))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("学习网线")
    plt.xlabel('训练样本数量')
    plt.ylabel(scoring)
    plt.grid()

    skipNum = 3
    plt.fill_between(train_sizes[skipNum:], 
                train_scores_mean[skipNum:] - train_scores_std[skipNum:],
                train_scores_mean[skipNum:] + train_scores_std[skipNum:], 
                alpha=0.1, color="b")
    plt.fill_between(train_sizes[skipNum:], 
                test_scores_mean[skipNum:] - test_scores_std[skipNum:],
                test_scores_mean[skipNum:] + test_scores_std[skipNum:], 
                alpha=0.1, color="g")

    plt.plot(train_sizes[skipNum:], train_scores_mean[skipNum:], 
            'o-', color="b", label="训练集")
    plt.plot(train_sizes[skipNum:], test_scores_mean[skipNum:], 
            'o-', color="r", label="测试集")

    plt.legend(loc="best")
    plt.show()


plot_learning_curve(mdl, X, y)

######################################################################
########  Part5、欠拟合解决方法：常规优化措施
######################################################################

# #####模型优化一：去除预测离群值

# 1.读取数据
filename = '回归分析.xlsx'
sheet = '计算机销售额'
df = pd.read_excel(filename, sheet_name=sheet)
print('cols=', df.columns.tolist())

# 2、准备数据
cols = ['失业率', '人均GNP', '教育开支']
target = '人均销售额'

# 属性选择
dfcorr = df.corr(method='spearman')
cond = np.abs(dfcorr[target]) > 0.3
cols = dfcorr[cond].index.tolist()
cols.remove(target)

X = df[cols]
y = df[target]

# 3、训练模型
mdl = LinearRegression()
mdl.fit(X, y)

# 4、评估模型
y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)
# 原始R2=0.5345, adjR2=0.4524

srPred = pd.Series( y - y_pred, index=df.index, name='误差')
df2 = pd.concat([df, srPred], axis=1)
print(df2)

# 5、优化一：去除离群值

# 查找预测离群值
resid = y - y_pred
std = np.std(resid)
cond = np.abs(resid) > 3*std
dfOutlier = df[cond]
print('预测离群值：\n',dfOutlier)

# 去除预测离群值
df.drop(index=dfOutlier.index, inplace=True)

# 再进行上述回归
X = df[cols]
y = df[target]
mdl = LinearRegression()
mdl.fit(X, y)

y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)

# 此时R2=0.741, adjR2=0.6924
# 模型质量由0.45上升到0.69


# #####模型优化二：增加非线性检验
# 1.读取数据
filename = '回归分析.xlsx'
sheet = '广告价格与销量'
df = pd.read_excel(filename, sheet_name=sheet)
print('cols=', df.columns.tolist())

# 2、准备数据
cols = ['价格', '广告费用']
target = '销量'

# 属性选择
dfcorr = df.corr(method='spearman')
cond = np.abs(dfcorr[target]) > 0.3
cols = dfcorr[cond].index.tolist()
cols.remove(target)

X = df[cols]
y = df[target]

# 3、训练模型
mdl = LinearRegression()
mdl.fit(X, y)

# 4、评估模型
y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)
# 原始R2=0.8337, adjR2=0.8317

# 5、优化二：新增非线性检验
df['价格*2'] = df['价格']**2
df['广告*2'] = df['广告费用']**2
cols = df.columns.tolist()
cols.remove(target)

X = df[cols]
y = df[target]

mdl = LinearRegression()
mdl.fit(X, y)

y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)
# 此时R2=0.8894, adjR2=0.8867

# 6、优化三：新增相互作用

df['价格*广告'] = df['价格'] * df['广告费用']
cols = df.columns.tolist()
cols.remove(target)

X = df[cols]
y = df[target]

mdl = LinearRegression()
mdl.fit(X, y)

y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)
# 此时R2=0.9939, adjR2=0.9937

# 以上优化二、优化三合起来，就是多项式回归
######################################################################
########  Part5、欠拟合解决方法：多项式回归
######################################################################

# 1.读取数据
filename = '回归分析.xlsx'
sheet = '身高年龄与体重'
df = pd.read_excel(filename, sheet_name=sheet)
print('cols=', df.columns.tolist())
cols = ['身高', '年龄']
target = '体重'

# 2、数据处理

# 属性选择
dfcorr = df.corr(method='spearman')
cond = np.abs(dfcorr[target]) > 0.3
cols = dfcorr[cond].index.tolist()
cols.remove(target)

X = df[cols]
y = df[target]

# 3、线性回归
mdl = LinearRegression()
mdl.fit(X, y)

# 4、评估模型
y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)

# 5、优化模型（优化数据、新增变量）
from sklearn.preprocessing import PolynomialFeatures

n = 4    #表示最高n次方
po = PolynomialFeatures(degree=n, include_bias=False)
fit_X = po.fit_transform(X)
print(fit_X.shape)


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

# n=4时，出现R2=1.0，说明过拟合
plot_learning_curve(mdl, fit_X, y)


# 6.应用模型
x = [[151, 12]]  #新数据集

# 注：要先进行多项式转换
fit_X = po.transform(x)
# 再利用模型预测
pred = mdl.predict(fit_X) 
print(pred)


######################################################################
########  Part6、正则项（过拟合解决方案）
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
# plot_learning_curve(mdl, fit_X, y)

# 5、优化模型（岭回归）

# ###########优化一，采用Ridge回归##########
from sklearn.linear_model import Ridge 

alpha = 5.0
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

# 超参优化（略）

# ###########优化二，采用Lasso回归##########
from sklearn.linear_model import Lasso

alpha = 5.0
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

# ###########优化三，采用ElasticNet回归##########
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

# 超参优化(略)

######################################################################
########  Part7、超参优化
######################################################################
# 四种方式：
# 1）手工遍历
# 2）交叉验证类RidgeCV/LassoCV/ElasticNetCV
# 3）网格搜索类GridSearchCV
# 4）随机搜索类RandomizedSearchCV
# ...

from scipy.linalg import hilbert

X = hilbert(10)

w = np.random.randint(2,10,10)
y = np.matrix(X) * np.matrix(w).T
y = np.array(y.T)[0]

# print(np.corrcoef(X))       #可知X数据集是多重共线性的

# 构造列标题：cols
cols = ['x%s'% x for x in range(1, X.shape[1]+1)]

# 先利用训练集进行
mdl = Ridge()      #默认alpha=1.0
mdl.fit(X, y)
y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)

############################
######## 1)超参优化：手工遍历
############################
from sklearn.model_selection import cross_val_score

print('\n======手工遍历，确定最优alpha值=============')

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
######## 2)超参优化：交叉验证
############################

############################
######## 岭回归 超参优化 RidgeCV
############################
# 带L2正则项
# 目标函数：||y - Xw||^2_2 + alpha * ||w||^2_2
from sklearn.linear_model import RidgeCV

print('\n======RidgeCV=============')

alphas = np.linspace(0.01, 10, 100)
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

# 评估模型（略）
# 保存模型（略）

############################
######## 套索回归 超参优化 LassoCV
############################
# 带L1正则项
# 目标函数：(1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

print('\n=======LassoCV==========')
from sklearn.linear_model import LassoCV

alphas = np.linspace(0.01, 10, 100)
mdl = LassoCV(alphas = alphas)
mdl.fit(X, y)

#返回最优的alpha参数,此时mdl即为最优模型
alpha = mdl.alpha_
print('最优alpha=', mdl.alpha_)
print('score = ', mdl.score(X, y))

sr = pd.Series(name='LassoCV系数',
                data = [mdl.intercept_] + mdl.coef_.tolist(),
                index = ['常数']+cols)
print(sr)
# 可看到很多系数为0，说明可以实现变量缩减。

# 评估模型（略）
# 保存模型（略）

############################
######## 弹性网络 超参优化 ElasticNetCV
############################
# 同时带L1和L2正则项
# 目标函数：(1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

print('\n=======ElasticNetCV==========')
from sklearn.linear_model import ElasticNetCV

alphas = np.linspace(0.01, 10, 100)
ratios = np.linspace(0, 1.0, 100)

mdl = ElasticNetCV(alphas=alphas, l1_ratio=ratios)
mdl.fit(X, y)

#返回最优参数,此时mdl即为最优模型
print('最优alpha=', mdl.alpha_)
print('最优l1_ration=', mdl.l1_ratio_)
print('score = ', mdl.score(X, y))

sr = pd.Series(name='ElasticNetCV系数',
                data = [mdl.intercept_] + mdl.coef_.tolist(),
                index = ['常数']+cols)
print(sr)


# 评估模型（略）
# 保存模型（略）

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

############################
######## 3)超参优化：网格搜索 GridSearchCV
############################
# 穷尽搜索最优超参
print('\n=======超参优化：GridSearchCV==========')
from sklearn.model_selection import GridSearchCV

# 构建参数字典
params = {'alpha':[1e-3, 1e-2, 1e-1, 1, 10, 100],
            'l1_ratio':[0.1,0.2,0.3, 1] }

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
print('\n=======超参优化：RandomizedSearchCV==========')
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


######################################################################
########  Part8：实战练习：线性回归
######################################################################
# 建立预测总费用的模型

# 1、读取数据
filename = 'Telephone.csv'
df = pd.read_csv(filename, encoding='gbk')
print(df.columns.tolist())

intCols = ['年龄','收入','家庭人数','开通月数']
catCols = ['居住地','婚姻状况','教育水平','性别']
target = '消费金额'

# 2、特征工程
# 1）寻找显著影响因素
import statsmodels.formula.api as smf
import statsmodels.stats.anova as smsa
# 仅做主效应检验
formula = target + ' ~ ' + \
            '+'.join(intCols) + '+' + \
            '+'.join(catCols)                

module = smf.ols(formula, df).fit()
dfRet = smsa.anova_lm(module)

# 取显著因子项
cond = dfRet['PR(>F)'] < 0.05
sr = dfRet[cond]
cols = sr.index.tolist()
print(cols)

# 筛选变量
for col in intCols:
    if col not in cols:
        intCols.remove(col)
for col in catCols:
    if col not in cols:
        catCols.remove(col)
# print(intCols, catCols)

# 2）类别型变量-->哑变量
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(dtype='int',drop='first',sparse=False)
X_ = enc.fit_transform(df[catCols])

TmpCols = []
for i in range(len(enc.categories_)):
    TmpCols += enc.categories_[i][1:].tolist()
# print('类别转化成哑变量后的标题：\n', TmpCols)

dfCats = pd.DataFrame(X_,
                index=df.index,
                columns=TmpCols)
# print(dfCats.head())

# 4）合成最终的数据集
dfInts = df[intCols]
dfCols = pd.concat([dfInts, dfCats], axis = 1)
print(dfCols.head())

cols = dfCols.columns.tolist()
X = dfCols
y = df[target]

# 3、训练模型
mdl = LinearRegression()
mdl.fit(X, y)

sr = pd.Series(name='实战回归系数',
            data=[mdl.intercept_]+mdl.coef_.tolist(),
            index= ['常数']+cols )
print('\n', sr)

# 4、评估模型
y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)

# 5、优化模型（略）


# 6、应用模型
df2 = df.iloc[1:10,:]

# 取连续变量值
dfInts = df2[intCols]

# 将类别变量转换catCols
x_ = enc.transform(df2[catCols])

# 合并
XX = np.concatenate([dfInts.values, x_], axis=1)

y_pred = mdl.predict(XX)
print(y_pred)

