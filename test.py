
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

    print(mts)

    return

# 1.读取数据
filename = '回归分析.xlsx'
sheet = '汽车季度销量'
df = pd.read_excel(filename, sheet)

# 2.特征工程/属性筛选

# 1)属性选择
catCols = ['季度']
intCols = ['GNP','失业率','利率']
target = '销量'

# 2）类别-->哑变量 (独热编码)
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(drop='first',sparse=False)
X_ = enc.fit_transform( df[catCols] )

dfCats = pd.DataFrame(
            data=X_,
            index=df.index,
            columns=enc.categories_[0][1:]
)
print(dfCats.head())

# 3）合并
dfCols = pd.concat([dfCats, df[intCols]], axis=1)

X = dfCols
y = df[target]

cols = dfCols.columns.tolist()

# 3.训练模型
from sklearn.linear_model import LinearRegression

mdl = LinearRegression()
mdl.fit(X, y)

# 打印回归系数/方程
sr = pd.Series(
        data=[mdl.intercept_] + mdl.coef_.tolist(),
        index=['常数'] + cols)
print(sr)

# 4.评估模型
y_pred = mdl.predict(X)
displayRegressionMetrics(y, y_pred, X.shape)

# 5.优化（略）
# 6.应用模型
# 1）预测
XX = [['第一季度', 2700, 4.5, 10.5]]
dfXX = pd.DataFrame( data=XX,columns=catCols+intCols)

# 先哑变量化
XX_ = enc.transform(dfXX[catCols])
dfXX_ = pd.DataFrame(data=XX_, columns=enc.categories_[0][1:])

# 合并
df2 = pd.concat([dfXX_, dfXX[intCols]], axis=1)

# 才预测
pred = mdl.predict(df2)

# 2）解读

# 3）保存
# 4）加载
