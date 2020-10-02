#-*- coding: utf-8 -*- 

########  本文件呈现 方差分析 相关知识点，包括：
# Part1.单因素方差分析(scipy.stats)
#   1)齐性检验scipy.stats.levene()    assert(p>0.05)
#   2)方差分析scipy.stats.f_oneway()  assert(p<0.05)

# Part2.单因素/多因素方差分析
# Part3.协方差分析
#   from statsmodels.formula.api import ols
#   from statsmodels.stats.anova import anova_lm

######################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

import statsmodels.formula.api as smf
import statsmodels.stats.anova as smsa

######################################################################
########  Part1.单因素方差分析:scipy.stats模块
######################################################################

# 1、读取数据集
filename = '相关性分析.xls'
sheetname = '位置与销量'
df = pd.read_excel(filename, sheetname)
print(df.columns.tolist())

col = '位置'
target = '月销量'

########### 单因素方差分析,使用scipy模块

# 1、方差齐性检验
# 分组
grouped = df.groupby(by=col)
groups = []

for name, grp in grouped:
    groups.append(grp[target])
# vals = df[col].unique()
# for val in vals:
#     groups.append(grouped.get_group(val)[target])

#### levene检验
from scipy import stats

threshold = 0.05
W,p = stats.levene(*groups)
# 相当于W,p = stats.levene(group1, group2, group3) #组必须是单列
if p < threshold:
    print('警告：因子不满足齐性检验要求！')
else:
    print('满足方差齐性检验。')

#### 2、作方差检验
# F, p = stats.f_oneway(group1, group2, group3)
F, p = stats.f_oneway(*groups)
print((F, p))

if p < threshold:
    print('因子={},因变量={}，有显著影响.'.format(col, target))
else:
    print('因子={},因变量={}，没有有显著影响！'.format(col, target))


#### 3、因素的最佳水平
# grouped = df.groupby(by='位置')
sr = grouped[target].mean()
idxmax = np.argmax(sr.values)
idxmin = np.argmin(sr.values)
print('最佳因素:argmax={},argmin={}'.format(sr.index[idxmax],sr.index[idxmin]))

# dfstastic = grouped[target].agg([np.size, np.sum, np.mean])
# dfstastic.columns = ['计数','求和','平均值']
# print(dfstastic)

#### 4、可视化
sr = grouped[target].mean()
sr.plot(kind='bar',title='平均销量')

# 最后，结论：
# 1）位置对销量有显著影响
# 2）位置放在前端时，销量最高

######################################################################
########  Part2.单因素/多因素方差分析：statsmodels.stats.anova模块
######################################################################
# 广义的线性回归模型

# 1、读取数据集
filename = '相关性分析.xls'
sheetname = '广告地区与销售额'
df = pd.read_excel(filename, sheetname)
print(df.columns.tolist())

catCols = ['广告形式','地区']
target = '销售额'

# 2、作主效应和交互项检验

import statsmodels.formula.api as smf
import statsmodels.stats.anova as smsa

# formula = '销量~广告形式'         #单因素
# formula = '销量~广告形式+地区'    #多因素（主效应）
# formula = '销量~广告形式*地区'    #多因素（主效应+交互项）

formula = '{} ~ {}'.format(
            target,
            '*'.join(catCols)
            )
module = smf.ols(formula, df).fit()
# print(module.summary())

print('方程R2={:.4}, adjR2={:.4}'.format(module.rsquared, module.rsquared_adj))

dfRet = smsa.anova_lm(module)  #返回DF对象
print(dfRet)

# 3、选择P<0.05的显著因子项
cond = dfRet['PR(>F)'] < 0.05
sr = dfRet[cond]
cols = list(sr.index)
print('\n显著影响因子：',cols)

# 4、因素及其组合的最佳水平
# cols = ['广告形式','地区','广告形式:地区']
for col in cols:
    # 统计
    byCols = col.split(':')
    groups = df.groupby(byCols)
    sr = groups[target].mean()
    print(sr)
    # 取最大值/最小值对应的最佳水平
    print('col={}, MAX={}[{:.4}], MIN={}[{:.4}]\n'.format(
                    byCols, 
                    sr.idxmax(), sr.max(),
                    sr.idxmin(), sr.min() )
                    )

    # 画图
    sr.plot(kind='bar',title='平均销量')
    plt.show()


# 5、多重比较
# 广告形式的多重组合是否会对销售额有显著的差异
from statsmodels.stats import multicomp as mc

col = '广告形式'
hsd = mc.pairwise_tukeyhsd(df[target], df[col])
print(hsd.summary())
# 可查看有显著差异的组合项

# 6、非饱和模型：去掉交互项，仅做主效应
formula = '{} ~ '.format(target)
formula += '+'.join(catCols)

module = smf.ols(formula, df).fit()
dfRet = smsa.anova_lm(module)  #返回DF对象
print(dfRet)
print(module.rsquared_adj)

######################################################################
########  封装函数：多因素方差分析:statsmodels.stats.anova模块
######################################################################
# 广义的线性回归模型

import statsmodels.formula.api as smf
import statsmodels.stats.anova as smsa

def catFeatureSelection(df, catCols, target, threshold=0.05):
    # 先做主效应检验
    formula = '{} ~ {}'.format(target, '+'.join(catCols))

    module = smf.ols(formula, df).fit()
    dfRet = smsa.anova_lm(module)

    # 取显著因子项
    cond = dfRet['PR(>F)'] < threshold
    cols = dfRet[cond].index.tolist()

    # 再做主效应和交互作用检验
    formula = '{} ~ {}'.format(target, '*'.join(catCols))

    module = smf.ols(formula, df).fit()
    dfRet = smsa.anova_lm(module)

    # 取显著因子项
    cond = dfRet['PR(>F)'] < threshold
    cols = dfRet[cond].index.tolist()

    # 找出最佳水平
    for col in cols:
        # 统计
        byCols = col.split(':')
        groups = df.groupby(byCols)
        sr = groups[target].mean()
        # print(sr)
        # 取最大值/最小值对应的最佳水平
        print('col={}, MAX={}[{:.4}], MIN={}[{:.4}]\n'.format(byCols, 
                        sr.idxmax(), sr.max(),
                        sr.idxmin(), sr.min() )
                        )
        
        # 可视化
        sr.plot(kind='bar', title='平均值')
        plt.show()

    return cols

# 1、读取数据集
filename = 'Telephone.csv'
dfTel = pd.read_csv(filename, encoding='gbk')
print(dfTel.columns.tolist())

catCols = ['居住地', '婚姻状况', '教育水平', '性别']
intCols = ['年龄', '收入','家庭人数','开通月数']
target = '消费金额'

eduLevel = ['初中','高中','大专','本科','研究生']
dfTel['教育水平'] = dfTel['教育水平'].astype('category').cat.set_categories(eduLevel)

# 2、单因素方差分析
cols = catFeatureSelection(dfTel, ['教育水平'], target)

# 3、多因素方差分析
cols = catFeatureSelection(dfTel, catCols, target)


######################################################################
########  Part3.协方差分析，statsmodels模块
######################################################################
# 自变量同时考虑类别型、数值型变量
# analysis of covariance

# 特征筛选函数
def featureSelection(df, catCols, intCols, target, threshold=0.05):
    # 先做主效应检验
    formula = target + ' ~ ' + \
                '+'.join(intCols) + '+' + \
                '+'.join(catCols)                

    module = smf.ols(formula, df).fit()
    dfRet = smsa.anova_lm(module)

    # 取显著因子项
    cond = dfRet['PR(>F)'] < threshold
    cols = dfRet[cond].index.tolist()

    # 对显著因子，再做主效应及交互项
    for col in catCols:
        if col not in cols:
            catCols.remove(col)
    for col in intCols:
        if col not in cols:
            intCols.remove(col)
    
    formula = '{} ~ {}+{}'.format(
                target,
                '*'.join(intCols),
                '*'.join(catCols) )

    module = smf.ols(formula, df).fit()
    dfRet = smsa.anova_lm(module)

    # 取显著因子项
    cond = dfRet['PR(>F)'] < threshold
    cols = dfRet[cond].index.tolist()
    print('\n显著影响因子：',cols)
    
    # 因素的最佳水平
    for col in cols:
        # 统计
        byCols = col.split(':')
        if byCols[0] in intCols:    #仅对类别型变量
            continue
        groups = df.groupby(byCols)
        sr = groups[target].mean()
        # print(sr)
        # 取最大值/最小值对应的最佳水平
        print('col={}, MAX={}[{:.4}], MIN={}[{:.4}]\n'.format(byCols, 
                        sr.idxmax(), sr.max(),
                        sr.idxmin(), sr.min() )
                        )
        
        # 主效应可视化
        sr.plot(kind='bar', title='平均值')
        plt.show()

    return cols

# 1、读取数据集
filename = 'Telephone.csv'
df = pd.read_csv(filename, encoding='gbk')
# print(df.columns.tolist())

catCols = ['居住地', '婚姻状况', '教育水平', '性别','电子支付']
intCols = ['年龄', '收入','家庭人数','开通月数']
target = '消费金额'

eduLevel = ['初中','高中','大专','本科','研究生']
dfTel['教育水平'] = dfTel['教育水平'].astype('category').cat.set_categories(eduLevel)

cols = featureSelection(dfTel, catCols, intCols, target)
