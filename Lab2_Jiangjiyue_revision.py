# -*- coding: utf-8 -*-

'''
Revision Code:
Name of Experiment: Lab2_Association Mining of Traditional Chinese Medicine Syndromes,
Author: Jiyue Jiang, Student ID: 17281093, Class ID: Medical Information 1707,
Supervised by Qiang Zhu, Beijing Jiaotong University,
Environment Configuration: Python3.7.7, PyCharm2019.3.3.
'''

# ����ģ��
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

# ���������ʾ����
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ȥ������
warnings.filterwarnings('ignore')


### ����Ԥ����
# �������ٰ����ߵĲ�����Ϣ���ݼ�
datafile = 'data.xls'
processedfile = 'processed_data.xls'

typelabel = {u'��������֤��ϵ��':'A', u'�ȶ��̽�֤��ϵ��':'B',  u'����ʧ��֤��ϵ��':'C',
             u'��Ѫ����֤��ϵ��':'D',  u'Ƣθ����֤��ϵ��':'E',  u'��������֤��ϵ��':'F'}

# ��Ҫ����������
k = 4

# ��ȡ���ݼ������洦�������ݼ�
data = pd.read_excel(datafile)
keys = list(typelabel.keys())
result = pd.DataFrame()

# չʾ���ݼ�
print("�������ٰ����ߵĲ�����Ϣ���ݼ��� \n")
print(data)

# ����
for i in range(len(keys)):
    # ����K-Means�����㷨�����о������
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

# index����
result = result.sort_index()

# ��������Excel����չʾ����������
result.to_excel(processedfile)
result = pd.read_excel(processedfile)
print("���������ݼ�չʾ���£�\n")
print(result)

'''
#������ɢ��
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
'''

# ����ֵ��ȡ��
columns = ["���̽׶�", "ת�Ʋ�λ", "ȷ����귢��ת��"]
data2 = data.drop(columns, axis = 1, inplace = False)
data2 = data2.sort_values(by = ['��������֤��ϵ��', '�ȶ��̽�֤��ϵ��', '����ʧ��֤��ϵ��', '��Ѫ����֤��ϵ��',
                                'Ƣθ����֤��ϵ��', '��������֤��ϵ��', "TNM����"])
data2 = data2.reset_index(drop = True)

# չʾ���������ݼ�
print("չʾ���������ݼ���\n")
print(data2)

# �����������б��浽�ı�,ȥ������
data2.to_csv("SecondData.txt", sep=',',index = False, header = None)

'''
# ��ģ���ݼ�
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

## ģ�ͽ���
'''
#���������ļ�
inputfile ='apriori.txt'
data2 = pd.read_csv(inputfile, header=None, dtype=object)
start = time.clock() # ��ʱ��ʼ
print(u'\nת��ԭʼ������0-1����')
ct = lambda x: Series(1, index = x[pd.notnull(x)])
b = map(ct, data2.as_matrix())
data3 = DataFrame(b).fillna(0)
end = time.time()
print (u'ת����ϣ���ʱ%s��' % (end-start))
del b
support = 0.06
confidence = 0.75
ms = '---'
# ��ʱ��ʼ
start = time.time()
print(u'\n��ʼ������������...')
find_rule(data3, support, confidence, ms)
end = time.clock() 
print (u'\n������ɣ���ʱ��%.2f��' % (end-start))
'''

# �ģ��
def data_generator(filename):
    def data_gen():
        with open(filename) as file:
            for line in file:
                yield tuple(k.strip() for k in line.split(','))
    return data_gen

# ���ļ����и�ʽ��
transactions = data_generator("SecondData.txt")

# ��ʱ��ʼ
start = time.clock()
print("Start searching for association rules. ")
itemsets, rules = apriori(transactions, min_support = 0.06, min_confidence = 0.75)
end = time.clock()
print("Search Autocomplete. The time is %0.2f second. " % (end - start))
print(rules)
rules = list(filter(lambda rule: len(rule.rhs) == 1 and rule.rhs[0][0] == 'H', rules))
for rule in rules:
    print(rule, rule.confidence, rule.support)


## Apriori�㷨
class Apriori:
    def __init__(self, min_support, min_confidence):
        # ��С֧�ֶ�
        self.min_support = min_support
        # ��С���Ŷ�
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
        # ������������
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
        # �����ŶȽ�������
        return sorted(result.items(), key=lambda x: -x[1][0])

# ��ʼ��ʱ
start = time.clock()
print("Start searching for association rules. ")
model = Apriori(0.06, 0.75)
rules = model.find_rules("SecondData.txt")
end = time.clock()
print("Search Autocomplete. The time is %0.2f second. " % (end - start))
pprint(rules)

## �Ⱦ��������ɢ��
# ��ȡ���ݼ�
data = pd.read_excel(datafile)

# �Ⱦ���䷨
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

# ����ֵ����
columns = ["���̽׶�", "ת�Ʋ�λ", "ȷ����귢��ת��"]

# �������ݼ���ɾ����Ӧ��
data3 = data.drop(columns, axis = 1, inplace = False)

# ����sort����
data3 = data3.sort_values(by = ['��������֤��ϵ��', '�ȶ��̽�֤��ϵ��', '����ʧ��֤��ϵ��', '��Ѫ����֤��ϵ��',
                                'Ƣθ����֤��ϵ��', '��������֤��ϵ��', "TNM����"])
data3 = data3.reset_index(drop = True)

#�����������б��浽�ı�,ȥ������
data3.to_csv("ThirdData.txt", sep = ',', index = False, header = None)

# ��ӡ���չʾ
print("��ӡ�����\n")
print(result)

# ��ӡdata3���չʾ
print("��ӡdThirdData�����\n")
print(data3)

# �����ݼ�
file = open("ThirdData.txt")

# ���й��������������б�
transactions = []
tup = []
for line in file.readlines():
    curLine = line.strip().split(",")
    curLine = tuple(map(str,curLine))
    transactions.append(curLine)

# ��ʼ��ʱ
start = time.clock()
print("Start searching for association rules. ")
itemsets, rules = apriori(transactions, min_support=0.06, min_confidence=0.75)
end = time.clock()
print("Search Autocomplete. The time is %0.2f second. " % (end - start))
print(rules)

# ����rules
rules = list(filter(lambda rule: len(rule.rhs) == 1 and rule.rhs[0][0] == 'H', rules))
for rule in rules:
    print(rule, rule.confidence, rule.support)

## ��Ƶ�ʷ�����ɢ��
# ��ȡ����
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
columns = ["���̽׶�", "ת�Ʋ�λ", "ȷ����귢��ת��"]

# ���ݴ���ɾ����Ӧ��
data4 = data.drop(columns, axis = 1, inplace = False)

# ��sort��������
data4 = data4.sort_values(by = ['��������֤��ϵ��', '�ȶ��̽�֤��ϵ��', '����ʧ��֤��ϵ��', '��Ѫ����֤��ϵ��',
                                'Ƣθ����֤��ϵ��', '��������֤��ϵ��', "TNM����"])

# ���½����е�����
data4 = data4.reset_index(drop = True)
data4.to_csv("FourthData.txt", sep = ',', index = False, header = None)

# ��ӡ���չʾ
print("��ӡ�����\n")
print(result)

# ���ݴ���
file = open("FourthData.txt")
transactions = []
tup = []
for line in file.readlines():
    curLine = line.strip().split(",")
    curLine = tuple(map(str,curLine))
    transactions.append(curLine)

# ��ʼ��ʱ
start = time.clock()
print("Start searching for association rules. ")
itemsets, rules = apriori(transactions, min_support = 0.06, min_confidence = 0.75)
end = time.clock()
print("Search Autocomplete. The time is %0.2f second. " % (end - start))
print(rules)

# rules���ݴ���
rules = list(filter(lambda rule: len(rule.rhs) == 1 and rule.rhs[0][0] == 'H',rules))
for rule in rules:
    print(rule, rule.confidence, rule.support)

# ��ȡ���ݣ���������
data = pd.read_excel(datafile)
result = pd.read_excel(processedfile)

'''
# ������ɢ��
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
'''

# ����ֵ��ȡ
columns = ["TNM����", "ת�Ʋ�λ", "ȷ����귢��ת��"]
data5 = data.drop(columns, axis = 1, inplace = False)
# sort����
data5 = data5.sort_values(by = ['��������֤��ϵ��', '�ȶ��̽�֤��ϵ��', '����ʧ��֤��ϵ��', '��Ѫ����֤��ϵ��',
                                'Ƣθ����֤��ϵ��', '��������֤��ϵ��', "���̽׶�"])
data5 = data5.reset_index(drop = True)
data5.to_csv("FifthData.txt", sep=',', index = False, header = None)

# �����ݼ���ʽ��
transactions = data_generator("FifthData.txt")

# ��ʼ��ʱ
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


## ��ҽ֤��ϵ����ת�Ʋ�λ�Ĺ�������
# ����ֵ��ȡ�����ݴ���
columns = ["TNM����", "���̽׶�", "ȷ����귢��ת��"]
data6 = data.drop(columns, axis = 1, inplace = False)
data6 = data6.sort_values(by = ['��������֤��ϵ��', '�ȶ��̽�֤��ϵ��', '����ʧ��֤��ϵ��', '��Ѫ����֤��ϵ��',
                                'Ƣθ����֤��ϵ��', '��������֤��ϵ��', "ת�Ʋ�λ"])
data6 = data6.reset_index(drop=True)
data6.to_csv("SixthData.txt", sep = ',', index = False, header = None)

# ��ӡdata6���ݼ�
print("��ӡSixthData���ݼ���\n")
print(data6)

# �����ݼ����и�ʽ������
transactions = data_generator("SixthData.txt")

# ��ʼ��ʱ
start = time.clock()
print("Start searching for association rules. ")
itemsets, rules = apriori(transactions, min_support = 0.08, min_confidence = 0.85)
end = time.clock()
print("Search Autocomplete. The time is %0.2f second. " % (end - start))
rules = list(filter(lambda rule: len(rule.rhs) == 1 and
                                 rule.rhs[0][0] == 'R', rules))
for rule in rules:
    print(rule, rule.confidence, rule.support)


## ��ҽ֤��ϵ����ȷ����귢��ת�ƵĹ�������
# ����ֵ��ȡ�����ݴ���
columns = ["TNM����", "���̽׶�", "ת�Ʋ�λ"]
data7 = data.drop(columns, axis = 1, inplace = False)
data7 = data7.sort_values(by = ['��������֤��ϵ��', '�ȶ��̽�֤��ϵ��', '����ʧ��֤��ϵ��', '��Ѫ����֤��ϵ��',
                                'Ƣθ����֤��ϵ��', '��������֤��ϵ��', "ȷ����귢��ת��"])
data7 = data7.reset_index(drop = True)
data7.to_csv("SeventhData.txt", sep = ',',index = False, header = None)

# ��ӡdata7���ݼ�
print("��ӡSeventhData���ݼ���\n")
print(data7)

# �����ݽ��и�ʽ������
transactions = data_generator("SeventhData.txt")

# ��ʼ��ʱ
start = time.clock()
print("Start searching for association rules. ")
itemsets, rules = apriori(transactions, min_support = 0.08, min_confidence = 0.85)
end = time.clock()
print("Search Autocomplete. The time is %0.2f second. " % (end - start))
rules = list(filter(lambda rule: len(rule.rhs) == 1 and
                                 rule.rhs[0][0] =='J',rules))
for rule in rules:
    print(rule, rule.confidence, rule.support)




