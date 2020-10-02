#-*- coding: utf-8 -*-

########  本文件实现回归模型实战，包括
# Part1、自定义管道类，实现数据预处理，及特征工程
# Part2、建立模型，优化模型
######################################################################

import pandas as pd
import numpy as np

from common import displayRegressionMetrics


######################################################################
########  Part1、定义管道类，实现数据预处理，及特征工程
######################################################################
# 自定义管道类要求：参考通用模块实现
# 1)必须实现fit和tansform两个函数
# 2)transform函数返回的处理后的X值，一般为numpy.ndarray类型数据

from common import MyFeaturePreprocessing

######################################################################
########  Part2、建立模型，优化模型
######################################################################
# 建立预测总费用的模型

# 1、读取数据
filename = 'Telephone.csv'
df = pd.read_csv(filename, encoding='gbk')
# print(df.columns.tolist())

caCols = ['居住地', '婚姻状况', '教育水平', '性别']
inCols = ['年龄', '收入','家庭人数','开通月数']
target = '消费金额'

# 2、数据预处理
cols = caCols + inCols

X = df[cols]
y = df[target]

# 3、数据建模
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

alpha = 1.2
pipe = Pipeline(steps=[
            ('mfp', MyFeaturePreprocessing()),
            ('mdl', Ridge(alpha=alpha))
            ])
pipe.fit(X, y)

# 打印回归模型
myFeature = pipe.named_steps['mfp']     #取管道中的对象
cols = myFeature.cols

# mdl = pipe.named_steps['mdl']
mdl = pipe['mdl']                       #取管道中的对象

sr = pd.Series(data=[mdl.intercept_] + mdl.coef_.tolist(),
            index=['常数'] + cols)
print(sr)

# 4.模型评估
y_pred = pipe.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)


# 5.超参优化
# pipe中的参数设置，参数格式：<estimator>__<parameter>
# 假定前面已经分好验证集
X_validate = X
y_validate = y

# ####手工遍历
alphas = np.linspace(1, 100, 20)
for alpha in alphas:
    pipe.set_params(mdl__alpha=0.8)
    pipe.fit(X_validate, y_validate)
    r2 = pipe.score(X_validate, y_validate)
    print('alpha={:.2f},  R2={:.4f}'.format(alpha, r2))

# ####网格搜索
from sklearn.model_selection import GridSearchCV

# params = dict(mdl_alpha=np.linspace(0.001, 1, 20))
params = {'mdl__alpha':np.linspace(0.001, 1, 20)}

grid = GridSearchCV(pipe, param_grid=params)
grid.fit(X_validate, y_validate)

print('最优参数：\n', grid.best_params_)

pipe = grid.best_estimator_
mdl = pipe['mdl']
sr = pd.Series(data=[mdl.intercept_] + mdl.coef_.tolist(),
            index=['常数'] + cols)
print(sr)

# 6、模型应用
# 1）训练集评估
y_pred = pipe.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)

# # 3）保存模型
# import joblib
# fname = 'out.mdl'
# joblib.dump(pipe, fname)

# # 4）加载模型
# pipeOut = joblib.load(fname)

# # 5）测试集评估
# pred = pipeOut.predict(X_test)
# displayRegressionMetrics(y_test, pred, X_test.shape)



