# -*- coding: utf-8 -*-

'''
Major Code:
Name of Experiment: Lab2_Association Mining of Traditional Chinese Medicine Syndromes,
Author: Jiyue Jiang, Student ID: 17281093, Class ID: Medical Information 1707,
Supervised by Qiang Zhu, Beijing Jiaotong University,
Environment Configuration: Python3.7.7, PyCharm2019.3.3.
'''

# 导入模块
import numpy as np
import pandas as pd
import random
import math
import time
import os
import sys
import warnings
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from efficient_apriori import apriori
from sklearn import preprocessing
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn import random_projection
from sklearn import manifold
from sklearn import decomposition
from pprint import pprint

# 解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 去除警告
warnings.filterwarnings('ignore')


### 数据预处理
# 三阴乳腺癌患者的病理信息数据集
datafile = 'data.xls'
processedfile = 'processed_data.xls'

typelabel = {u'肝气郁结证型系数':'A', u'热毒蕴结证型系数':'B',  u'冲任失调证型系数':'C',
             u'气血两虚证型系数':'D',  u'脾胃虚弱证型系数':'E',  u'肝肾阴虚证型系数':'F'}

# 需要聚类的类别数
k = 4

# 读取数据集并保存处理后的数据集
data = pd.read_excel(datafile)
keys = list(typelabel.keys())
result = pd.DataFrame()

# 展示数据集
print("三阴乳腺癌患者的病理信息数据集： \n")
print(data)

# 聚类
for i in range(len(keys)):
    # 调用K-Means聚类算法，进行聚类分析
    print(u"The clustering of %s is in progress. " % keys[i])
    kmodel = KMeans(n_clusters = k, n_jobs = 4)
    kmodel.fit(data[[keys[i]]].iloc[:,:].values)
    r1 = pd.DataFrame(kmodel.cluster_centers_, columns = [typelabel[keys[i]]])
    r2 = pd.Series(kmodel.labels_).value_counts()
    r2 = pd.DataFrame(r2, columns = [typelabel[keys[i]] + 'n'], dtype = 'int')
    r = pd.concat([r1, r2], axis = 1).sort_values(typelabel[keys[i]])
    r.index = [1, 2, 3, 4]
    r[typelabel[keys[i]]] = r[typelabel[keys[i]]].rolling(2).mean()
    r[typelabel[keys[i]]][1] = 0.0
    result = result.append(r.T)

# index排序
result = result.sort_index()

# 保存结果到Excel表并且展示处理后的数据
result.to_excel(processedfile)
result = pd.read_excel(processedfile)
print("处理后的数据集展示如下：\n")
print(result)

#数据离散化
for i in range(len(keys)):
    for j in range(data.shape[0]):
        if data.iloc[j,i] <= result[2][chr(ord(65)+i)]:
            data.iloc[j,i]  = chr(ord(65)+i) +'1'
        elif data.iloc[j,i]  > result[2][chr(ord(65)+i)] and data.iloc[j,i]  <= result[3][chr(ord(65)+i)]:
            data.iloc[j,i]  = chr(ord(65)+i) +'2'
        elif data.iloc[j,i]  > result[3][chr(ord(65)+i)] and data.iloc[j,i]  <= result[4][chr(ord(65)+i)]:
            data.iloc[j,i]  = chr(ord(65)+i) +'3'
        else:
            data.iloc[j,i]  = chr(ord(65)+i) +'4'

# 特征值提取，
columns = ["病程阶段", "转移部位", "确诊后几年发现转移"]
data2 = data.drop(columns, axis = 1, inplace = False)
data2 = data2.sort_values(by = ['肝气郁结证型系数', '热毒蕴结证型系数', '冲任失调证型系数', '气血两虚证型系数',
                                '脾胃虚弱证型系数', '肝肾阴虚证型系数', "TNM分期"])
data2 = data2.reset_index(drop = True)

# 展示处理后的数据集
print("展示处理后的数据集：\n")
print(data2)

# 不将索引序列保存到文本,去除列名
data2.to_csv("SecondData.txt", sep=',',index = False, header = None)

'''
# 建模数据集
data_cut = DataFrame(columns = data.columns[:6])
types = ['A', 'B', 'C', 'D', 'E', 'F']
num = ['1', '2', '3', '4']
for i in range(len(data_cut.columns)):
    value = list(data.iloc[:,i])
    bins = list(result[(2*i):(2*i+1)].values[0])
    bins.append(1)
    names = [str(x)+str(y) for x in types for y in num]
    group_names = names[4*i:4*(i+1)]
    cats = pd.cut(value,bins,labels=group_names,right=False)
    data_cut.iloc[:,i] = cats
data_cut.to_excel('apriori.xlsx')
data_cut.head()
'''

## 模型建立
'''
#输入事务集文件
inputfile ='apriori.txt'
data2 = pd.read_csv(inputfile, header=None, dtype=object)
start = time.clock() # 计时开始
print(u'\n转换原始数据至0-1矩阵')
ct = lambda x: Series(1, index = x[pd.notnull(x)])
b = map(ct, data2.as_matrix())
data3 = DataFrame(b).fillna(0)
end = time.time()
print (u'转换完毕，用时%s秒' % (end-start))
del b
support = 0.06
confidence = 0.75
ms = '---'
# 计时开始
start = time.time()
print(u'\n开始搜索关联规则...')
find_rule(data3, support, confidence, ms)
end = time.clock() 
print (u'\n搜索完成，用时：%.2f秒' % (end-start))
'''

# 搭建模型
def data_generator(filename):
    def data_gen():
        with open(filename) as file:
            for line in file:
                yield tuple(k.strip() for k in line.split(','))
    return data_gen

# 将文件进行格式化
transactions = data_generator("SecondData.txt")

# 计时开始
start = time.clock()
print("Start searching for association rules. ")
itemsets, rules = apriori(transactions, min_support = 0.06, min_confidence = 0.75)
end = time.clock()
print("Search Autocomplete. The time is %0.2f second. " % (end - start))
print(rules)
rules = list(filter(lambda rule: len(rule.rhs) == 1 and rule.rhs[0][0] == 'H', rules))
for rule in rules:
    print(rule, rule.confidence, rule.support)


## Apriori算法
class Apriori:
    def __init__(self, min_support, min_confidence):
        # 最小支持度
        self.min_support = min_support
        # 最小置信度
        self.min_confidence = min_confidence
    def count(self, filename = 'apriori.txt'):
        self.total = 0
        items = {}
        with open(filename) as f:
            for l in f:
                self.total += 1
                for i in l.strip().split(','):
                    if i in items:
                        items[i] += 1.
                    else:
                        items[i] = 1.
        self.items = {i: j / self.total for i, j in items.items() if j / self.total > self.min_support}
        self.item2id = {j: i for i, j in enumerate(self.items)}
        # 制作布尔矩阵
        self.D = np.zeros((self.total, len(items)), dtype = bool)
        with open(filename) as f:
            for n, l in enumerate(f):
                for i in l.strip().split(','):
                    if i in self.items:
                        self.D[n, self.item2id[i]] = True
    def find_rules(self, filename = 'apriori.txt'):
        self.count(filename)
        rules = [{(i,): j for i, j in self.items.items()}]
        l = 0
        while rules[-1]:
            rules.append({})
            keys = sorted(rules[-2].keys())
            num = len(rules[-2])
            l += 1
            for i in range(num):
                for j in range(i + 1, num):
                    if keys[i][:l - 1] == keys[j][:l - 1]:
                        _ = keys[i] + (keys[j][l - 1],)
                        _id = [self.item2id[k] for k in _]
                        support = 1. * sum(np.prod(self.D[:, _id], 1)) / self.total
                        if support > self.min_support:
                            rules[-1][_] = support
        result = {}
        for n, relu in enumerate(rules[1:]):
            for r, v in relu.items():
                for i, _ in enumerate(r):
                    x = r[:i] + r[i + 1:]
                    confidence = v / rules[n][x]
                    if confidence > self.min_confidence:
                        result[x + (r[i],)] = (confidence, v)
        # 按置信度降序排列
        return sorted(result.items(), key=lambda x: -x[1][0])

# 开始计时
start = time.clock()
print("Start searching for association rules. ")
model = Apriori(0.06, 0.75)
rules = model.find_rules("SecondData.txt")
end = time.clock()
print("Search Autocomplete. The time is %0.2f second. " % (end - start))
pprint(rules)

## 等距离分箱离散化
# 读取数据集
data = pd.read_excel(datafile)

# 等距分箱法
result = pd.DataFrame()
for i in range(len(keys)):
    tmp = data[keys[i]]
    length = (max(tmp) - min(tmp)) / 4
    side = [min(tmp) + length, min(tmp) + length * 2, min(tmp) + length * 3, max(tmp)]
    count = [0, 0, 0, 0]
    for j in range(data.shape[0]):
        if data.iloc[j, i] < side[0]:
            data.iloc[j, i] = chr(ord('A')+i) + '1'
            count[0] = count[0] + 1
        elif data.iloc[j, i] >= side[0] and data.iloc[j, i]  < side[1]:
            data.iloc[j, i] = chr(ord('A') + i) + '2'
            count[1] = count[1] + 1
        elif data.iloc[j,i] >= side[1] and data.iloc[j, i]  < side[2]:
            data.iloc[j,i] = chr(ord('A') + i) +'3'
            count[2] = count[2] + 1
        else:
            data.iloc[j, i] = chr(ord('A') + i) +'4'
            count[3] = count[3] + 1
    result = result.append(pd.DataFrame(side, columns = [typelabel[keys[i]]]).T)
    result = result.append(pd.DataFrame(count, columns = [typelabel[keys[i]] + 'n']).T)

# 特征值处理
columns = ["病程阶段", "转移部位", "确诊后几年发现转移"]

# 处理数据集，删除对应列
data3 = data.drop(columns, axis = 1, inplace = False)

# 进行sort排序
data3 = data3.sort_values(by = ['肝气郁结证型系数', '热毒蕴结证型系数', '冲任失调证型系数', '气血两虚证型系数',
                                '脾胃虚弱证型系数', '肝肾阴虚证型系数', "TNM分期"])
data3 = data3.reset_index(drop = True)

#不将索引序列保存到文本,去除列名
data3.to_csv("ThirdData.txt", sep = ',', index = False, header = None)

# 打印结果展示
print("打印结果：\n")
print(result)

# 打印data3结果展示
print("打印dThirdData结果：\n")
print(data3)

# 打开数据集
file = open("ThirdData.txt")

# 进行关联分析的数据列表
transactions = []
tup = []
for line in file.readlines():
    curLine = line.strip().split(",")
    curLine = tuple(map(str,curLine))
    transactions.append(curLine)

# 开始计时
start = time.clock()
print("Start searching for association rules. ")
itemsets, rules = apriori(transactions, min_support=0.06, min_confidence=0.75)
end = time.clock()
print("Search Autocomplete. The time is %0.2f second. " % (end - start))
print(rules)

# 处理rules
rules = list(filter(lambda rule: len(rule.rhs) == 1 and rule.rhs[0][0] == 'H', rules))
for rule in rules:
    print(rule, rule.confidence, rule.support)

## 等频率分箱离散化
# 读取数据
data = pd.read_excel(datafile)
result = pd.DataFrame()
for i in range(len(keys)):
    tmp = list(data[keys[i]])
    tmp.append(1)
    tmp.append(1)
    tmp = np.array(tmp)
    tmp.sort()
    tmp = tmp.reshape([4, -1])
    side = [tmp[0][-1], tmp[1][-1], tmp[2][-1], tmp[3][-3]]
    count = [0, 0, 0, 0]
    for j in range(data.shape[0]):
        if data.iloc[j,i] <= side[0]:
            data.iloc[j,i]  = chr(ord('A') + i) +'1'
            count[0] = count[0] + 1
        elif data.iloc[j,i]  > side[0] and data.iloc[j,i] <= side[1]:
            data.iloc[j,i]  = chr(ord('A') + i) +'2'
            count[1] = count[1] + 1
        elif data.iloc[j,i]  > side[1] and data.iloc[j,i] <= side[2]:
            data.iloc[j,i]  = chr(ord('A') + i) +'3'
            count[2] = count[2] + 1
        else:
            data.iloc[j,i]  = chr(ord('A') + i) +'4'
            count[3] = count[3] + 1
    result = result.append(pd.DataFrame(side, columns = [typelabel[keys[i]]]).T)
    result = result.append(pd.DataFrame(count, columns = [typelabel[keys[i]] + 'n']).T)
columns = ["病程阶段", "转移部位", "确诊后几年发现转移"]

# 数据处理，删除对应列
data4 = data.drop(columns, axis = 1, inplace = False)

# 对sort进行排序
data4 = data4.sort_values(by = ['肝气郁结证型系数', '热毒蕴结证型系数', '冲任失调证型系数', '气血两虚证型系数',
                                '脾胃虚弱证型系数', '肝肾阴虚证型系数', "TNM分期"])

# 重新建立行的索引
data4 = data4.reset_index(drop = True)
data4.to_csv("FourthData.txt", sep = ',', index = False, header = None)

# 打印结果展示
print("打印结果：\n")
print(result)

# 数据处理
file = open("FourthData.txt")
transactions = []
tup = []
for line in file.readlines():
    curLine = line.strip().split(",")
    curLine = tuple(map(str,curLine))
    transactions.append(curLine)

# 开始计时
start = time.clock()
print("Start searching for association rules. ")
itemsets, rules = apriori(transactions, min_support = 0.06, min_confidence = 0.75)
end = time.clock()
print("Search Autocomplete. The time is %0.2f second. " % (end - start))
print(rules)

# rules数据处理
rules = list(filter(lambda rule: len(rule.rhs) == 1 and rule.rhs[0][0] == 'H',rules))
for rule in rules:
    print(rule, rule.confidence, rule.support)

# 读取数据，处理数据
data = pd.read_excel(datafile)
result = pd.read_excel(processedfile)

# 数据离散化
for i in range(len(keys)):
    for j in range(data.shape[0]):
        if data.iloc[j,i] <= result[2][chr(ord('A') + i)]:
            data.iloc[j,i] = chr(ord('A')+i) + '1'
        elif data.iloc[j,i] > result[2][chr(ord('A') + i)] and\
                data.iloc[j,i] <= result[3][chr(ord('A') + i)]:
            data.iloc[j,i] = chr(ord('A') + i) + '2'
        elif data.iloc[j,i] > result[3][chr(ord('A') + i)] and\
                data.iloc[j,i] <= result[4][chr(ord('A') + i)]:
            data.iloc[j,i] = chr(ord('A') + i) + '3'
        else:
            data.iloc[j,i] = chr(ord('A') + i) + '4'

# 特征值提取
columns = ["TNM分期", "转移部位", "确诊后几年发现转移"]
data5 = data.drop(columns, axis = 1, inplace = False)
# sort排序
data5 = data5.sort_values(by = ['肝气郁结证型系数', '热毒蕴结证型系数', '冲任失调证型系数', '气血两虚证型系数',
                                '脾胃虚弱证型系数', '肝肾阴虚证型系数', "病程阶段"])
data5 = data5.reset_index(drop = True)
data5.to_csv("FifthData.txt", sep=',', index = False, header = None)

# 将数据集格式化
transactions = data_generator("FifthData.txt")

# 开始计时
start = time.clock()
print("Start searching for association rules. ")
itemsets, rules = apriori(transactions, min_support = 0.10, min_confidence = 0.90)
end = time.clock()
print("Search Autocomplete. The time is %0.2f second. " % (end - start))
rules = list(filter(lambda rule: len(rule.rhs) == 1 and
                                 len(rule.lhs) == 2 and
                                 rule.rhs[0][0] == 'S', rules))
for rule in rules:
    print(rule, rule.confidence, rule.support)


## 中医证型系数与转移部位的关联分析
# 特征值提取和数据处理
columns = ["TNM分期", "病程阶段", "确诊后几年发现转移"]
data6 = data.drop(columns, axis = 1, inplace = False)
data6 = data6.sort_values(by = ['肝气郁结证型系数', '热毒蕴结证型系数', '冲任失调证型系数', '气血两虚证型系数',
                                '脾胃虚弱证型系数', '肝肾阴虚证型系数', "转移部位"])
data6 = data6.reset_index(drop=True)
data6.to_csv("SixthData.txt", sep = ',', index = False, header = None)

# 打印data6数据集
print("打印SixthData数据集：\n")
print(data6)

# 将数据集进行格式化处理
transactions = data_generator("SixthData.txt")

# 开始计时
start = time.clock()
print("Start searching for association rules. ")
itemsets, rules = apriori(transactions, min_support = 0.08, min_confidence = 0.85)
end = time.clock()
print("Search Autocomplete. The time is %0.2f second. " % (end - start))
rules = list(filter(lambda rule: len(rule.rhs) == 1 and
                                 rule.rhs[0][0] == 'R', rules))
for rule in rules:
    print(rule, rule.confidence, rule.support)


## 中医证型系数与确诊后几年发现转移的关联分析
# 特征值提取和数据处理
columns = ["TNM分期", "病程阶段", "转移部位"]
data7 = data.drop(columns, axis = 1, inplace = False)
data7 = data7.sort_values(by = ['肝气郁结证型系数', '热毒蕴结证型系数', '冲任失调证型系数', '气血两虚证型系数',
                                '脾胃虚弱证型系数', '肝肾阴虚证型系数', "确诊后几年发现转移"])
data7 = data7.reset_index(drop = True)
data7.to_csv("SeventhData.txt", sep = ',',index = False, header = None)

# 打印data7数据集
print("打印SeventhData数据集：\n")
print(data7)

# 将数据进行格式化处理
transactions = data_generator("SeventhData.txt")

# 开始计时
start = time.clock()
print("Start searching for association rules. ")
itemsets, rules = apriori(transactions, min_support = 0.08, min_confidence = 0.85)
end = time.clock()
print("Search Autocomplete. The time is %0.2f second. " % (end - start))
rules = list(filter(lambda rule: len(rule.rhs) == 1 and
                                 rule.rhs[0][0] =='J',rules))
for rule in rules:
    print(rule, rule.confidence, rule.support)




