#-*- coding: utf-8 -*- 

########  本文件呈现 卡方检验 相关知识点，包括：

# Part1.卡方检验scipy.stats.chisquare
#   卡方值chi、显著性p
# Part2.交叉表-可视化

######################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

######################################################################
########  Part1.卡方检验
######################################################################
#直接对计数和期望值的计算
# from scipy import stats
# obs = [102, 102, 96, 105, 95, 100]
# exp = [100, 100, 100, 100, 100, 100]
# chi2, p = stats.chisquare(f_obs = obs, f_exp = exp)    


# 1、读取数据集
filename = 'Telephone.csv'
df = pd.read_csv(filename, encoding='gbk')
# print(df.columns.tolist())

eduLevel = ['初中','高中','大专','本科','研究生']
df['教育水平'] = df['教育水平'].astype('category').cat.set_categories(eduLevel)

# 判断学历与套餐是否有关系
catCol1 = '套餐类型'
catCol2 = '流失'

def chi2Test(df, catCol1, catCol2, threshold=0.05):
    # 计数值
    grouped = df.groupby(by=[catCol1,catCol2])
    obs = grouped[catCol2].count()
    obs.name = '计数值'
    # print(obs)
    
    # 期望值
    total = obs.sum()
    exps = pd.Series(index=obs.index)
    for tc, lv in obs.index:
        val = obs[tc].sum() * obs[:,lv].sum() /total
        # exps.append(val)
        exps[(tc, lv)] = val
    exps.name = '期望值'
    # print(exps)

    # 卡方检验
    chi2, p = stats.chisquare(f_obs=obs, f_exp = exps) 
    print('\nchi2={:.4},p={:.4}'.format(chi2, p))

    if p < threshold:
        print('因子[{}]与[{}]有显著相关性。'.format(catCol1, catCol2))
    else:
        print('因子[{}]与[{}]没有显著相关性!'.format(catCol1, catCol2))

    return p < threshold


ret = chi2Test(df, catCol1, catCol2)

# 遍历：卡方检验
catCols = ['居住地', '婚姻状况', '教育水平', '性别', '电子支付', '套餐类型']
target = '流失'

cols = []
for col in catCols:
    ret = chi2Test(df, col, target)
    if ret:
        cols.append(col)
print('\n有显著影响的因子有：\n', cols)

######################################################################
########  Part2.交叉表-可视化
######################################################################

# 交叉表
df2 = pd.pivot_table(df, index='套餐类型', 
            columns='流失', values='UID',
            aggfunc='count')

# 复式柱状图
df2.plot(kind='bar')

# 堆积柱状图
df2.plot(kind='bar', stacked=True)

# 堆积百分比柱状图
total = df2.sum(axis=1)
for col in df2.columns:
    df2[col] = df2[col]/total   #转化为行百分比
df2.plot(kind='bar', stacked=True)

# 对影响因素作百分比堆积柱状图
for col in cols:
    df2 = pd.pivot_table(df, 
                index=col,
                columns=target,
                values='UID',
                aggfunc='count')
    
    #转化为行百分比
    total = df2.sum(axis=1)
    for cl in df2.columns:
        df2[cl] = df2[cl]/total
    
    df2.plot(kind='bar', stacked=True)


