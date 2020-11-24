
import pandas as pd
from pandas.core.arrays import sparse

# 1.读取数据
filename = 'Telephone.csv'
df = pd.read_csv(filename, encoding='gbk')
# print(df.columns)

# 2.特征工程

# 1）字段标识
catCols = ['居住地', '婚姻状况', '教育水平', '性别', '电子支付']
intCols = ['年龄', '收入', '家庭人数', '开通月数']
target = '消费金额'

# 2)哑变量化
# 类别变量-->哑变量
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse=False, drop='first')
X_ = enc.fit_transform( df[catCols] )
# print(X_.shape)

cols = []
for cats in enc.categories_:
    cols.extend(cats[1:])
# print(cols)

dfCats = pd.DataFrame(
        data=X_,
        columns=cols
)
# print(dfCats)

# 3)数值型变量（略）

# 4）合并（类别+数值）
X = pd.concat( [dfCats, df[intCols] ], axis=1)
y = df[target]

cols = X.columns.tolist()

# 3.训练模型
# 1)训练
from sklearn.linear_model import LinearRegression

mdl = LinearRegression()
mdl.fit(X, y)

# 2）打印回归系数
sr = pd.Series(
        data=[mdl.intercept_] + mdl.coef_.tolist(),
        index=['常数'] + cols
)
print(sr)

# 4.评估模型
r2 = mdl.score(X, y)
print('判定系数R^2=', r2)

# 5.优化模型（略）
# 6.应用模型
# 1）解读模型（略）
# 2）保存模型
import joblib
file = 'out.mdl'
joblib.dump(mdl, file)

# 3)加载模型
mdl_reg = joblib.load(file)

# 4）预测
X0 = [['上海','未婚','高中','女','No',39,78,1,41]]
X0_cats = [['上海','未婚','高中','女','No']]
X0_ints = [[39,78,1,41]]

# 哑变量化
X0_ = enc.transform(X0_cats)

# 合并
import numpy as np
XX = np.concatenate( [X0_, X0_ints], axis=1)

# 预测
pred = mdl.predict(XX)
print(pred)

# from sklearn.linear_model import Ridge, Lasso,ElasticNet
