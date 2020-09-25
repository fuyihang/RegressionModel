#-*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


filename = '回归分析.xlsx'
sheet = '销售额'
df = pd.read_excel(filename, sheet)

cols = ['办公费用', '营销费用']
target = '销售额'
X = df[cols].values
y = df[target].values


