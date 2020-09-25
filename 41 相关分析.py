#-*- coding: utf-8 -*- 

########  本文件呈现 相关分析 相关知识点，包括：
#   Part1.简单相关系数: r=sr1.corr(sr2, method)
#       1)pearson相关
#       2)spearman相关
#       3)kendalltau相关
#   Part2.相关系数矩阵: dfcorr = df.corr(method)
#   Part3.相关系数及显著性:
#       1)scipy.stats.pearsonr
#       2)scipy.stats.spearmanr
#       3)scipy.stats.kendalltau
#   Part4.正态性检验：
#       1)scipy.stats.normaltest
#       2)scipy.stats.shapiro
#       3)scipy.stats.kstest
#   Part5.显著性判断:stats.t.pdf
#       1)T分布检验
#       2)Z分布检验
#   Part6.偏相关分析：相关系数,p显著性
######################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats


# 1、读取数据集
filename = '相关性分析.xls'
sheetname = '腰围与体重'
df = pd.read_excel(filename, sheetname)

# 2、可视化观察
plt.scatter(df['腰围'], df['体重'])
plt.xlabel('腰围')
plt.ylabel('体重')
plt.show()

######################################################################
########  Part1. 简单相关分析
######################################################################
# 计算两个数值型变量的相关系数

# 计算相关系数
r = df['腰围'].corr(df['体重'], method='pearson')
print('腰围和体重的相关系数r=', r)

# method取值：pearson, kendall, spearman

######################################################################
########  Part2. 相关系数矩阵
######################################################################
# 一次性实现多个变量间的相关系数

# 计算相关系数矩阵
dfcorr = df.corr()  #默认 method='pearson'
print(dfcorr)

r = dfcorr.loc['腰围', '体重']
# r = dfcorr['腰围']['体重']
print('腰围和体重的相关系数r=', r)

######################################################################
########  Part3. 相关系数，及显著性判断
######################################################################
# 计算相关系数，并同时判断显著性
from scipy import stats

# 计算相关系数，及显著性水平
r, p = stats.pearsonr(df['腰围'], df['体重'])
# r, p = stats.spearmanr(df['腰围'], df['体重'])
# r, p = stats.kendalltau(df['腰围'], df['体重'])
print('r={:.4},p={:.4}'.format(r, p))

if p < 0.05:
    print('两个变量存在线性相关性')

######################################################################
########  Part4. 正态性检验
######################################################################
# 判断变量是否服从正态分布

# 正态性检验，有几种方法：
#   K-S检验
#   S-W检验 shapiro-Wilk
#   normaltest检验
from scipy import stats

sr = df['腰围']
threshold = 0.05        #显著性水平

# 方法一：使用normaltest函数检验
# 比较常用，通过偏度和峰度来判断
k2, p = stats.normaltest(sr)
if p > threshold:
    print('变量"{}"服从正态分布。'.format(sr.name))
else:
    print('变量"{}"不是正态分布的！'.format(sr.name))

# 方法二：使用Shapiro-Wilk检验
# 适合小样本量（<5000)的正态性检验

W,p = stats.shapiro(sr)
if p > threshold:
    print('变量"{}"服从正态分布。'.format(sr.name))
else:
    print('变量"{}"不是正态分布的！'.format(sr.name))

# 方法三：使用K-S检验
# 适合大样本量（>5000)的正态性检验

u = sr.mean()
std = sr.std()
print('mean={},std={}'.format(u, std))

D, p = stats.kstest(sr, 'norm', (u,std))
if p > threshold:
    print('变量"{}"服从正态分布。'.format(sr.name))
else:
    print('变量"{}"不是正态分布的！'.format(sr.name))


######################################################################
########  Part5. 显著性检验（手工实现）
######################################################################

# 1）利用t统计量检验（pearson相关系数检验）
r = df['腰围'].corr(df['体重'])
n = len(df)
if abs(r) == 1:
    T = 0
else:
    T = abs(r)*np.sqrt( (n-2) / (1-r**2) )
p = stats.t.pdf(T, df=n-2)      #此处df表示自由度
print(r, p)
if p < 0.05:
    print('两个变量存在线性相关')


# 2)利用Z统计量检验（spearman相关系数检验）
r = df['腰围'].corr(df['体重'], method='spearman')
n = len(df)
if r == 1:
    Z = 0
else:
    Z = abs(r)*np.sqrt(n-1)
p = stats.norm.pdf(Z)
print(p)
if p < 0.05:
    print('两个变量存在线性相关')

######################################################################
########  Part6. 偏相关分析
######################################################################
# 排除控制变量后，两变量之间的相关性
# 一阶偏相关
def partialCorr(df, xCol, yCol, zCol, threshold=0.05):
    method = 'pearson'

    rxy = df[xCol].corr(df[yCol], method)
    rxz = df[xCol].corr(df[zCol], method)
    ryz = df[yCol].corr(df[zCol], method)
    # print(rxy, rxz, ryz)

    # 偏相关系数
    rxy_z = (rxy - rxz*ryz) /np.sqrt( (1-rxz**2) * (1-ryz**2) )
    print('rxy_z={:.4}'.format(rxy_z))

    # 显著性
    n = len(df); q = 1
    if abs(rxy_z) ==1:
        T = 0
    else:
        T = rxy_z * np.sqrt( (n-q-2)/(1-rxy_z**2) )
    p = stats.t.pdf(T, df=n-q-2)
    # print('T={:.4},p={:.4}'.format(T, p))

    if p < threshold:
        print("x={},y={},z={},存在显著的线性偏相关".format(xCol, yCol ,zCol))
    else:
        print("x={},y={},z={},没有显著的线性偏相关！".format(xCol, yCol ,zCol))

    return rxy_z, p<threshold   #返回相关系数，及显著性p值

x = '腰围'
y = '体重'
z = '脂肪比重'     #控制变量
ret = partialCorr(df, x, y, z)

#################练习####################

# 1、读取数据集
filename = 'Telephone.csv'
dfTel = pd.read_csv(filename, encoding='gbk')

intCols = ['年龄', '收入', '开通月数']
target = '消费金额'

# 2、相关矩阵
dfcorr = dfTel[intCols+[target]].corr(method='spearman')
print(dfcorr)

# 3、偏相关
# 排除开通月数后，判断年龄、收入对消费金额的影响
intCols = ['年龄', '收入']
z = '开通月数'          #控制变量
target = '消费金额'

for col in intCols:
    ret = partialCorr(dfTel, col, target, z)

# 相关函数
    # DataFrame.corr(method='pearson', min_periods=1)
    # method : {‘pearson’, ‘kendall’, ‘spearman’} or callable
    #   pearson : standard correlation coefficient
    #   kendall : Kendall Tau correlation coefficient
    #   spearman : Spearman rank correlation
    #   callable: callable with input two 1d ndarrays
    #   and returning a float .. versionadded:: 0.24.0
    # min_periods : int, optional
    #   Minimum number of observations required per pair of columns to have a valid result. 
    #   Currently only available for pearson and spearman correlation
