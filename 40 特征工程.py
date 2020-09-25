#-*- coding: utf-8 -*- 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
########  本文件实现变量处理功能，包括：
# Part1. 变量转换（常规派生、变换）
# Part2. 变量标准化、正则化
# Part3. 变量离散化 （数值型-->类别型）
# Part4. 数字化     （类别型-->数值型）
# Part5. 哑变量化   （类别型-->k个哑变量）
# Part6. 变量合并(PCA, FA)
# Part7. sklearn.feature_selection包
#########################################
"""

########  关于变量降维：
# Part1、按变量本身特性来筛选
# 1）如果变量的缺失比例过大
# 2）去除方差较小的变量、比例不均衡的变量
    # 对所有变量中最大比例样本对应的比例大于等于80%的变量予以剔除
# 3）
# 二、按自变量与目标变量的相关性来筛选
# 参考前面的相关分析、方差分析、卡方检验

# 三、
# 1）SelectKBest():按得分，方差分析F值
# 2）SelectPercentile():按得分从高到低，选取百分比
# 3）SelectFpr():按FPR来选择,alpha
# 4）GenericUnivariateSelect
# https://www.cnblogs.com/feffery/p/8808398.html
# https://blog.csdn.net/qq_34840129/article/details/82927156
# https://www.jianshu.com/p/bd48f67db7c6
######################################################################

######################################################################
########  Part1. 去除比例不均衡的变量
######################################################################

# 1、读取数据集
filename = 'Telephone.csv'
dfTel = pd.read_csv(filename, encoding='gbk')


# 只取数值型变量
catCols = ['居住地', '婚姻状况', '教育水平', '性别']
intCols = ['年龄', '收入','家庭人数','开通月数']
target = '消费金额'

X = dfTel[intCols]

from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=0.8*(1-0.2))
X_ = sel.fit_transform(X)
print('特征选择后：\n', X_)

from scipy import stats

# 2、选择最恰当的相关系数计算公式
# 数值型变量的选择函数
def intFeatureSelection(df, intCols, target, 
                pthreshold=0.05, rthreshold=0.3):
    cols = []
    for col in intCols:
        r,p = stats.spearmanr(df[col], df[target])
        print('{}-{}:r={:.2f},p={:.6f}'.format(col, target, r, p))
        if (p < pthreshold) and (r > rthreshold):
            cols.append(col)
    print('对“{}”有影响的变量有：\n{}'.format(target, cols))
    return cols

cols = intFeatureSelection(dfTel, intCols, target)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



filename = 'Telephone.csv'
df = pd.read_csv(filename, encoding='gbk')
print(df.columns.tolist())

######################################################################
########  Part1.变量转换（常规派生、变换）
######################################################################
## 变量派生，根据旧变量，生成新的变量
# 比如基于出生日期，计算年龄

## 变量变换：将原变量进行函数计算，转换成其它值
X = df['年龄']
X_ = 1/X
X_ = np.abs(X) + 2
X_ = np.log10(X)

# 小数定标标准化，缩放到[-1, 1]范围
# 变换公式：Decimal scaling y=(X/10的k次方) (k确保maX|y|<1)
X_ = X/10**np.ceil(np.log10(np.abs(X).maX()))

# 对数Logistic模式 y = 1/(1+e^(-X))
X_ = 1/(1+np.eXp(-X))


#####=======分位数转换QuantileTransformer========================
# 转换为均匀分布（按照累积概率密度函数），转换后范围为[0,1]
from sklearn.preprocessing import QuantileTransformer
trans = QuantileTransformer()

cols = ['年龄','收入']
X_ = trans.fit_transform(df[cols])
print(X_)

#####=======PowerTransformer============
# 变换为高斯分布 Gaussian distribution
from sklearn.preprocessing import PowerTransformer
trans = PowerTransformer(method='yeo-johnson')

cols = ['年龄','收入']
X_ = trans.fit_transform(df[cols])
print(X_)

# method取值：
#    'yeo-johnson'
#    'box-cox'


## 自定义一个函数进行转换
from sklearn.preprocessing import FunctionTransformer
trans = FunctionTransformer(np.log10)

X_ = trans.transform(X)


######################################################################
########  Part2.变量规范化
######################################################################

#####=======StandardScaler============
### z-score标准化：y = (X-mean)/std
# 标准化：将不同规模和量纲的数据处理，缩放到相同的数据区间和范围，以减少规模、特征、分布差异等对模型的影响。
# 做法：将数据转换为标准正态分布（均值=0，标准差= 1），这样可利用正态分布的特征，
# 一种去中心化方法，会改变原数据分布，不适合于稀疏数据处理。

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_ = ss.fit_transform(X)
print(X_)

# # 可以查看其中重要属性
# ss.n_samples_seen_    #样本数量
# ss.mean_      #平均值
# ss.var_       #方差
# ss.scale_     #缩放比例，即标准差

# X_test_scaled = ss.transform(X_test)  #还可以对测试集进行转换
# X0 = ss.inverse_transform(X_)     #或者还原原始数据

# 其他类似，不再赘述！

#####=======MinMaXScaler============
from sklearn.preprocessing import MinMaxScaler
# 数据缩小到指定范围(min, max)，默认(0,1)，保持原有分布形态
# 变换公式：y=(X-min)/(maX-min)
# 作用：对于方差小的变量可以增强其稳定性。

#####=======MaXAbsScaler============
from sklearn.preprocessing import MaxAbsScaler
# 数据缩小到[-1, 1]之间的数，保持原有分布形态
# 变换公式：y = X/abs(maX(X))
# 适合于稀疏数据的处理（包括稀疏的CSR-Compressed Sparse Row行压缩 或CSC-Compressed Sparse Column列压缩矩阵）

#####=======RobustScaler============
from sklearn.preprocessing import RobustScaler
# 变换公式：y = (X-median)/IQR
# 如果数据有许多异常值，且想保留其离群特征，则可使用RobustScaler

#####=======Normalization============
##正则化：目的是将数据缩放到单位范数(每个字段的范数为1)
# 先计算其p-范数，再除以该范数。变换后的变量其范数等于1.
# 参数说明:
    # p-范数的计算公式：||X||p=(|X1|^p+|X2|^p+...+|Xn|^p)^1/p，
    # l1,l2范式分别指p=1或p=2的结果
    # norm：可以为l1、l2或maX，默认为l2
    # 若为l1时，旧值除以所有特征值的绝对值之和
    # 若为l2时，旧值除以所有特征值的平方之和的平均根
    # 若为maX时，旧值除以样本中特征值最大的值

from sklearn.preprocessing import Normalizer
# 注意aXis默认为1，所以一定要指定aXis=0
norm = Normalizer(norm='l1')
X_ = norm.fit_transform(df[cols])
print(X_)


######################################################################
######## Part3. 变量离散化/分箱（数值型-->类别型）
######################################################################

### 二值离散化{0，1}
# 大于threshold都标记为1，小于等于threshold的都标记为0.
from sklearn.preprocessing import Binarizer

cols =['年龄','收入']

est = Binarizer(threshold=50)
X_ = est.fit_transform(df[cols ])
print(X_)

### 多值离散化，分箱数n_bins
from sklearn.preprocessing import KBinsDiscretizer

est = KBinsDiscretizer(n_bins=5, encode='ordinal')    #0~n_bins-1
X_ = est.fit_transform(df[cols])
print(X_)

# KBinsDiscretizer
    # (n_bins=5, encode=’onehot’, strategy=’quantile’)
    #n_bins : int or array-like, shape (n_features,) (default=5)
    #   分箱数，不能小于2
    # encode : {‘onehot’, ‘onehot-dense’, ‘ordinal’}, (default=’onehot’)
        #编码结果的方法
        #onehot：one-hot编码，返回稀疏矩阵sparse array.
        #onehot-dense：one-hot编码，返回密集矩阵dense array.
        #ordinal：返回分箱标识（整数值）.
    # strategy : {‘uniform’, ‘quantile’, ‘kmeans’}, (default=’quantile’)
        # 分箱宽度的策略Strategy used to define the widths of the bins.
        # uniform：等宽，所有箱都有相同的宽度
        # quantile：等频，所有箱都有相同的点数
        # kmeans：K均值分箱，每个箱中点的距离接近

    # 如果要给每一个变量不同的分箱数，配置n_bins为数组
    # est = KBinsDiscretizer(n_bins=[3,2,2], encode='ordinal')    
    #默认是左闭右开区间
    # feature 1: [−∞, -1), [-1,2),[2,∞]
    # feature 2: [−∞, 5), [5, ∞)
    # feature 3: [−∞, 14), [14, ∞)


# 1、等宽离散化
# 注：df['年龄']的取值范围是[18, 47]

#返回每个索引的对应的分段，分成4个段，NaN自成一段
bins = 4        #分成4段
lbls = ['儿童','青年','中年','老年']
sr = pd.cut(df['年龄'], bins)                #默认的分段名是一个区间
sr = pd.cut(df['年龄'], bins, labels=lbls)   #给每段取名称
df['年龄分段'] = sr

print(sr.value_counts())       #返回各分箱的计数

# 2、自定义间隔,可以实现非等距分组。
# 注：切割区间一般是左开右闭区间，如(0,13],(13,30],(30,50],(50,100]
# 注意最小值和最大值的设定，不在此范围内的被归属于NaN
bins = [17, 22, 30, 40, 47]
sr = pd.cut(df['年龄'], bins, labels=lbls)
# 自定义间隔,要注意左边界应该要小于最小值

bins = [-np.inf, 22, 30, 40, np.inf]  #np.inf为无穷大
sr = pd.cut(df['年龄'], bins, labels=lbls)

# 其余均值-标准差分组，类似为自定义分组实现

# 3、等频离散化（每等分里面的数据量基本一样）
# 指定组数
bins = 4        #相当于分成4等分 [0, .25, .5, .75, 1]
sr = pd.qcut(df['年龄'], bins, labels=lbls)

# 也可自定义百分位切割点
#自定义每等分的切割点的百分位，加上最后1这个点
w = [1.0*i/bins for i in range(bins+1)]
#生成切割点w = [0, .25, .5, .75, 1]

sr = pd.qcut(df['年龄'], w, labels=lbls)

# 如果想知道等频离散化的切割点，即各分位数
des = df['年龄'].describe(percentiles = w)
w = des[4:4+bins+1]

# 4、K均值离散化

from sklearn.cluster import KMeans
col = '年龄'
df = df[[col]]

df_zs = 1.0* (df - df.mean())/df.std() 

k = 4   #假定分成4类
kmodel = KMeans(n_clusters = k)
kmodel.fit(df_zs)

#输出聚类中心, 一维中心
centers = pd.Series(kmodel.cluster_centers_[:,0])
centers.sort_values(inplace=True)
#相邻两项求中点，作为边界点
w = centers.rolling(2).mean().iloc[1:]  #去掉第一个空值
#把首末边界点加上
w = [df_zs[col].min()*(1-1e-10)] + list(w) + [df_zs[col].max()]

sr = pd.cut(df_zs[col], w, labels = range(k)) #分箱

# 相关函数
    # pandas.cut(x, bins, right=True, labels=None, 
    #               retbins=False, precision=3, include_lowest=False)
    # bins  1）整数，则表示分箱数； 
    #       2）序列，表示自定义非均匀宽度
    # right=True, 左开右闭区间
    # labels=None,每段的标签，取值：
    #       None 默认为分段的区间
    #       False，则返回整数填充的类别(0起始)
    #       序列,则自定义分段名称

    # pandas.qcut(x, q, labels=None, retbins=False, precision=3, duplicates='raise')
    # x : 1d ndarray or Series
    # q : integer or array of quantiles
    #       1)整数表示几分位。比如4表示四位分
    #       2）自定义分数序列.[0, .25, .5, .75, 1]
    # labels : array or boolean, default None
    # Used as labels for the resulting bins. Must be of the same length as the resulting bins. If False, return only integer indicators of the bins.
    # retbins : bool, optional
    #   Whether to return the (bins, labels) or not. 
    #   Can be useful if bins is given as a scalar.
    # precision : int, optional
    # The precision at which to store and display the bins labels
    # duplicates : {default ‘raise’, ‘drop’}, optional
    # If bin edges are not unique, raise ValueError or drop non-uniques.
    # New in version 0.20.0.


######################################################################
######## Part4. 数字化（类别型-->数值型）
######################################################################

### 类别型-->整数（整数值）
# 编码成0~k-1的整数
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder(dtype='int')

cols = ['居住地', '性别']
X_ = enc.fit_transform(df[cols])
print(X_)
# 返回类别值列表,按数值顺序排序的
print(enc.categories_)

# 只能对单个列表数字化
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()

cols = ['居住地']
X_ = enc.fit_transform(df[cols])
print(X_)
print(enc.classes_)


# OrdinalEncoder
    # (categories=’auto’, dtype=<class ‘numpy.float64’>)
    #categories : ‘auto’ or a list of lists/arrays of values.
    #   Categories (unique values) per feature:
    #   ‘auto’ : 自动决定类别值
    #   list :指定期望的类别列表
    #dtype : number type, default np.float64 输出的类型


# LabelEncoder
    # 编码成0~k-1的整数 

######################################################################
######## Part5. 哑变量
######################################################################
# 哑变量/二值虚拟变量

# 1、类别-->哑变量
# K分类，生成K个，或者K-1个哑变量
from sklearn.preprocessing import LabelBinarizer

# K>2时，生成K个哑变量
enc = LabelBinarizer()
col = '居住地'
X_ = enc.fit_transform(df[col])
print(X_)
print(enc.classes_)

# K=2时，只生成1个哑变量
col = '性别'
X_ = enc.fit_transform(df[col])
print(X_)
print(enc.classes_)

# 2、类别-->哑变量
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(dtype='int', drop='first', sparse=False)

cols = ['居住地','性别']
X_ = enc.fit_transform(df[cols])
print(X_)
print(enc.categories_)

# 3、数值-->哑变量
# 分箱哑变量化，生成n_bins个哑变量
from sklearn.preprocessing import KBinsDiscretizer
est = KBinsDiscretizer(n_bins=5, encode='onehot-dense') 
cols = ['年龄','收入']
X_ = est.fit_transform(df[cols])
print(X_)

print('每个特征的箱数：', est.n_bins_)
print('分箱的边界：', est.bin_edges_)


# LabelBinarizer
    # (neg_label=0, pos_label=1, sparse_output=False)
    # 二值化标签 one-vs-all
    # neg_label : int (default: 0)
    #   负类编码为此值
    # pos_label : int (default: 1)
    #   正类编码为此值
    # sparse_output : boolean (default: False)
    #   True：返回sparse CSR format.


# OneHotEncoder
    # (categorical_features=None, categories=None, drop='first',
    #   dtype=<class 'numpy.float64'>, handle_unknown='error',
    #   n_values=None, sparse=True)
    # categories ,指定类别标题顺序
    # categorical_features = 'all'，
    #     这个参数指定了对哪些特征进行编码，默认对所有类别都进行编码。
    #     也可以自己指定选择哪些特征，通过索引或者 bool 值来指定
    # dtype=<class ‘numpy.float64’> 表示编码数值格式，默认是浮点型
    # sparse=True 表示编码的格式，默认为 True，即为稀疏的格式，
    #       指定 False 则就不用 toarray() 了
    # handle_unknown=’error’，其值可以指定为 "error" 或者 "ignore"，即如果碰到未知的类别，是返回一个错误还是忽略它。
    # 

######################################################################
########  Part7. 特征选择
######################################################################
# 选择前N个得分最高的重要的特征

from sklearn.feature_selection import SelectKBest       #选择前K个重要特征
from sklearn.feature_selection import SelectPercentile  #选择前百分比个重要特征

# sklearn提供的两类函数
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2

filename = 'Telephone.csv'
df = pd.read_csv(filename, encoding='gbk')
print(df.columns.tolist())

################# 示例一：回归问题
targetCol = '消费金额'
y = df[targetCol]

#只考虑连续变量
cols = ['年龄', '收入','家庭人数','开通月数']  
X = df[cols]

k = 2
skb = SelectKBest(score_func=f_regression, k=k)
X_ = skb.fit_transform(X, y)

print(X_)
print('scores=\n', skb.scores_)
print('pvalues=\n', skb.pvalues_)

# 可视化特征重要性
plt.bar(cols, skb.scores_)
plt.title('影响[{}]的因素'.format(targetCol))
plt.show()

# 查看选出的是哪几个特征变量,得分最高的几个变量
sr = pd.Series(skb.scores_, index=cols)
sr.sort_values(inplace=True, ascending=False)
print(sr.index[:k].tolist())


################# 示例二：分类问题（chi2）
targetCol = '套餐类型'
y = df[targetCol]

#只考虑类别变量
cols = ['居住地', '婚姻状况', '教育水平', '性别']

# 可惜一定要将类型型变量进行数字化
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder(dtype='int')
X = enc.fit_transform(df[cols])
# print(X_)

k = 2
skb = SelectKBest(score_func=chi2, k=k)
X_ = skb.fit_transform(X, y)

print(X_)
print('scores=\n', skb.scores_)
print('pvalues=\n', skb.pvalues_)

# 可视化特征重要性
plt.bar(cols, skb.scores_)
plt.title('影响[{}]的因素'.format(targetCol))
plt.show()

# 查看选出的是哪几个特征变量,得分最高的几个变量
sr = pd.Series(skb.scores_, index=cols)
sr.sort_values(inplace=True, ascending=False)
print(sr.index[:k].tolist())

################# 示例三：分类问题

targetCol = '流失'
y = df[targetCol]

# 考虑所有类型的变量
catCols = ['居住地', '婚姻状况', '教育水平', '性别', '套餐类型']
intCols = ['年龄', '收入','家庭人数','开通月数']
cols = intCols+catCols  

# 可惜一定要将类型型变量进行数字化
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder(dtype='int')
X_ = enc.fit_transform(df[catCols])
# print(X_)
X = np.concatenate([X_, df[intCols]], axis=1)

k = 5
skb = SelectKBest(score_func=f_classif, k=k)
X_ = skb.fit_transform(X, y)

# print(X_)
print('scores=\n', skb.scores_)
print('pvalues=\n', skb.pvalues_)

# 可视化特征重要性
plt.bar(cols, skb.scores_)
plt.title('影响[{}]的因素'.format(targetCol))
plt.show()

# 查看选出的是哪几个特征变量,得分最高的几个变量
sr = pd.Series(skb.scores_, index=cols)
sr.sort_values(inplace=True, ascending=False)
print(sr.index[:k].tolist())

# 或者
idxs = list(np.argsort(skb.scores_))  #返回从小到大的序号
print(idxs[-k:])                      #返回后面k个序号
sr = pd.Series(skb.scores_, index=cols)
print(sr.index[idxs[-k:]].tolist())

# score_func参数取值
    # 分类
    # f_classif: ANOVA F-value between label/feature for classification tasks. 
    # mutual_info_classif: Mutual information for a discrete target. 
    # chi2: Chi-squared stats of non-negative features for classification tasks. 

    # 回归
    # f_regression: F-value between label/feature for regression tasks. 
    # mutual_info_regression: Mutual information for a continuous target. 

    # 其它
    # SelectPercentile: Select features based on percentile of the highest scores. 
    # SelectFpr: Select features based on a false positive rate test. 
    # SelectFdr: Select features based on an estimated false discovery rate. 
    # SelectFwe: Select features based on family-wise error rate. 
    # GenericUnivariateSelect: Univariate feature selector with configurable mode.


# 变量转换
from sklearn.preprocessing import FunctionTransformer   #自定义转换
from sklearn.preprocessing import PowerTransformer      #正态分布
from sklearn.preprocessing import QuantileTransformer   #均匀分布

# 常用标准化
from sklearn.preprocessing import StandardScaler    #z-score标准化：y = (X-mean)/std
from sklearn.preprocessing import MaxAbsScaler      #y = X/abs(maX(X))  [-1,+1]
from sklearn.preprocessing import MinMaxScaler      #y=(X-min)/(maX-min) [0, 1]
from sklearn.preprocessing import RobustScaler      #y = (X-median)/IQR
#正则化
from sklearn.preprocessing import Normalizer

# 数值型变量 --> 离散化
from sklearn.preprocessing import Binarizer         #离散化，二分{0,1}，<=阈值为0，大于阈值为1
from sklearn.preprocessing import KBinsDiscretizer  
# 分箱离散化，bins分箱
# 'onehot-dense'分箱，生成bins个哑变量化
# 'ordinal'分箱，并转化成数字
# 'onehot'  和onehot-dense相同，但要使用toarray()转换

# 类别型变量 --> 数字化
from sklearn.preprocessing import LabelEncoder      #变成0~k-1的数字,只能对单列处理
from sklearn.preprocessing import OrdinalEncoder    #变成0~k-1的数字,可以对多列处理

# 类别型变量 --> 哑变量化
from sklearn.preprocessing import LabelBinarizer    #生成k个二值虚拟变量;当k=2时，则生成一个哑变量。只能对单列处理
# from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder     #生成K或K-1个哑变量

# 变量降维
from sklearn.decomposition import PCA

# 特征选择
from sklearn.feature_selection import SelectKBest       #选择前K个重要特征
from sklearn.feature_selection import SelectPercentile  #选择前百分比个重要特征
