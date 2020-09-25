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
dfTel = pd.read_csv(filename, encoding='gbk')
print(dfTel.columns.tolist())

eduLevel = ['初中','高中','大专','本科','研究生']
dfTel['教育水平'] = dfTel['教育水平'].astype('category').cat.set_categories(eduLevel)

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

# 判断学历与套餐是否有关系
col1 = '套餐类型'
col2 = '教育水平'
ret = chi2Test(dfTel, col1, col2)

# 遍历：卡方检验
catCols = ['居住地', '婚姻状况', '教育水平', '性别', '电子支付', '套餐类型']
target = '流失'

cols = []
for col in catCols:
    ret = chi2Test(dfTel, col, target)
    if ret:
        cols.append(col)
print('\n有显著影响的因子有：\n', cols)

######################################################################
########  Part2.交叉表-可视化
######################################################################

def plotBar(df, indexCol, typeCol, valCol,
        stacking=False, percentile=False,
        title='堆积柱状图',dataLabel=True):
    """\
    画复式柱状图/堆积柱状图/堆积百分比柱状图
    """

    # 分类汇总
    df2 = pd.pivot_table(df, index=indexCol, 
                columns=typeCol, values=valCol,
                aggfunc='count')

    # 加上这段，就是百分比
    if percentile:
        df2['Total'] = df2.sum(axis=1)  #行汇总：按列汇总
        for i in range(len(df2.columns)-1):
            df2.iloc[:,i] = df2.iloc[:,i ] /df2.iloc[:, -1]
    
        df2.drop('Total', axis=1,inplace=True)
    
    if stacking:    #堆积图
        xlables = df2.index
        xpos = np.arange(len(xlables))
        btm = pd.Series([0]*len(df2), index=df2.index)

        for col in df2.columns:
            rects = plt.bar(xpos, df2[col], bottom=btm,
                        label=col, tick_label=xlables)
            if dataLabel:
                for j, rect in enumerate(rects):
                    x = rect.get_x() + rect.get_width()/3
                    y = rect.get_height()/3 + btm[j]
                    if percentile:
                        txt = '{:.0%}'.format(rect.get_height())
                    else:
                        txt = '{}'.format(rect.get_height())
                    plt.text(x, y, txt)

            btm += df2[col]
    else:   #复式图
        #取1为间隔，所以总宽为0.8，共要并列3个图例
        xlables = df2.index
        total_width, n = 0.8, len(df2.columns)
        width = total_width / n
        xpos = np.arange(len(xlables))
        xpos = xpos - (total_width - width) / 2

        for i in range(n):
            rects = plt.bar(xpos+i*width, 
                        df2.iloc[:, i],  width=width, 
                        tick_label=xlables, label=df2.columns[i])

            if dataLabel==True:
                for rect in rects:
                    x = rect.get_x()
                    height = np.round(rect.get_height(),2)
                    plt.text(x, 1.01*height, '{}'.format(height))

    plt.yticks([])  # 隐藏坐标轴刻度值
    plt.xlabel(indexCol)
    plt.ylabel(valCol)
    plt.title(title)
    plt.legend()
    plt.show()


indexCol = '套餐类型'
typeCol = '流失'
valCol = 'UID'
plotBar(dfTel, indexCol, typeCol, valCol)
plotBar(dfTel, indexCol, typeCol, valCol, stacking=True)
plotBar(dfTel, indexCol, typeCol, valCol,stacking=True, percentile=True)

indexCol = '教育水平'
typeCol = '流失'
valCol = 'UID'
plotBar(dfTel, indexCol, typeCol, valCol)
plotBar(dfTel, indexCol, typeCol, valCol, stacking=True)
plotBar(dfTel, indexCol, typeCol, valCol,stacking=True, percentile=True)

indexCol = '电子支付'
typeCol = '流失'
valCol = 'UID'
plotBar(dfTel, indexCol, typeCol, valCol)
plotBar(dfTel, indexCol, typeCol, valCol, stacking=True)
plotBar(dfTel, indexCol, typeCol, valCol,stacking=True, percentile=True)
