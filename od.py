
#########################################################
# Using pyOD KNN sample as basis for this program
#########################################################

from __future__ import division
from __future__ import print_function

import os
import sys
import csv
import numpy as np
from scipy import stats
import matplotlib.pyplot as plot

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyod.models.knn import KNN
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
from pyod.models.combination import aom, moa, average, maximization
from pyod.utils.utility import standardizer


#n_clf = 25
#k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 500, 1000, 3000, 5000, 10000]
n_clf = 20
k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
#n_clf = 6
#k_list = [80, 100, 200, 300, 400, 600]

# -------------------------------------------
# Preprocess train file for python processing by
# tr '\r' '\n' < File > File_1
# -------------------------------------------
CS = []
Mem = []
training_data = "train.txt"
test_data = "test.txt"
#training_data = "DLA_do_object_1.csv"
#test_data = "DLA_OBJ_ITER_1.csv"

with open(training_data, "r") as file:
    # Skip non data lines
    for _ in range(1):
        next(file)
    readCSV = csv.reader(file, delimiter=',')
    for column in readCSV:
        # remove JVM code segment information from SOURCE IP 0033:
        # print(int(column[0], 16), int(column[1][5:], 16))
        # code segment is key, memory region where data was read is value
        code_seg_tr = int(column[1][5:], 16)
        memory_region_tr = int(column[0], 16)
        CS.append(code_seg_tr)
        Mem.append(memory_region_tr)
        #point = [code_seg_tr, memory_region_tr]

    print('TRAIN MAX Mem = {}, CS = {}'.format(max(Mem), max(CS)))

    mem_max = np.full(len(Mem), max(Mem))
    CS_max  = np.full(len(CS), max(CS))
    # Normalize data so that large numbers become sensible
    CS_n = np.true_divide(CS, CS_max)
    Mem_n = np.true_divide(Mem, mem_max)
    diff_code_data = np.subtract(CS, Mem)
    d_train = np.c_[CS_n, Mem_n]

file.close()
"""
# Data Validation
#
for i in range(100):
    print(diff_code_data)
# Data sanity validation
for i in range(100):
    print(Mem[i], d_train[i])
"""

# -------------------------------------------
# Preprocess test file for python processing by
# tr '\r' '\n' < File > File_1
# -------------------------------------------
CS_t = []
Mem_t = []
with open(test_data, "r") as file:
    # Skip non data lines
    for _ in range(1):
        next(file)
    readCSV = csv.reader(file, delimiter=',')
    for column in readCSV:
        # remove JVM code segment information from SOURCE IP 0033:
        # print(int(column[0], 16), int(column[1][5:], 16))
        # code segment is key, memory region where data was read is value
        code_seg_tst = int(column[1][5:], 16)
        memory_region_tst = int(column[0], 16)
        CS_t.append(code_seg_tst)
        Mem_t.append(memory_region_tst)
        #point = [code_seg_tst, memory_region_tst]

    
    print('TEST MAX Mem = {}, CS = {}'.format(max(Mem), max(CS)))

    mem_t_max = np.full(len(Mem_t),max(Mem_t))
    CS_t_max  = np.full(len(CS_t), max(CS_t))
    # Normalize data so that large numbers become sensible
    CS_t_n = np.true_divide(CS_t,CS_t_max)
    Mem_t_n = np.true_divide(Mem_t,mem_t_max)
    d_test = np.c_[CS_t_n, Mem_t_n]

file.close()
# Data sanity validation
"""
for i in range(100):
    print(Mem_t[i], d_test[i])
"""
print('Data lengths test = {} max = {}-{}, train = {}, max = {}-{}'.format(len(d_test), max(Mem_t_n), max(CS_t_n), len(d_train), max(Mem_n), max(CS_n)))

#exit()
feature_1_train = d_train[:, [0]].reshape(-1,1)
feature_2_train = d_train[:, [1]].reshape(-1,1)
feature_1_test  = d_test[:, [0]].reshape(-1,1)
feature_2_test  = d_test[:, [1]].reshape(-1,1)

#scatter plot
plot.scatter(feature_1_train,feature_2_train)
plot.scatter(feature_1_test,feature_2_test)
plot.xlabel('Normalized Code Segment:Instruction Pointers')
plot.ylabel('Normalized Memory Access Addresses')
plot.yscale('linear')
plot.savefig('Feature.png', dpi=300, bbox_inches='tight')
#plot.show()
#exit()

d_train_norm, d_test_norm = standardizer(d_train, d_test)
train_scores = np.zeros([d_train.shape[0], n_clf])
test_scores = np.zeros([d_test.shape[0], n_clf])
#########################################
# All data accesses going to upper memory regions are classified as 
# outliers... 
# Examples are
# 0xFFFFFFFF8152A7F0
# 0xFFFFFFFF8100BB80
# 0x00007FF1741B0968
# 0x00007FEF9C254968
# 0x00007FC8F1FBF628
# 0x00007FC87AC46368
# 0x00007FC87AC46360
#########################################

dy_train=[]
dy_test=[]
#DMAX
#27333556840

for i in range(len(Mem)):
    #if Mem[i] > 34299489320:
    if Mem_t[i] > 24299489320:
        dy_train.append(1.)
    else:
        dy_train.append(0.)

#dy_train_f = np.zeros(len(d_train) - 570)
#dy_train_t = np.ones(570)
#dy_train = np.r_[dy_train_f, dy_train_t]


for i in range(len(Mem_t)):
    #if Mem_t[i] > 34299489320:
    if Mem_t[i] > 24299489320:
        dy_test.append(1.)
    else:
        dy_test.append(0.)

print('Combining {n_clf} kNN detectors'.format(n_clf=n_clf))
clf = []

point = 2000 # you can pick any data point for this analysis (0..50000)
point_analysis = []
clf_name = 'KNN'
for i in range(n_clf):
    k = k_list[i]
    print(i,k)
    clf_r = KNN(n_neighbors=k, method='largest')
    clf.append(clf_r)
    clf[i].fit(d_train_norm)
    train_scores[:, i] = clf[i].decision_scores_
    test_scores[:, i] = clf[i].decision_function(d_test_norm)
    point_analysis.append(clf[i].decision_scores_[point])

clf_s = KNN()
clf_s.fit(d_train)

print(point_analysis)
#
# https://towardsdatascience.com/time-series-of-price-anomaly-detection-13586cd5ff46
# Susan Li's Blog on Time series anomaly detection
#
fig, ax = plot.subplots(figsize=(10,6))
ax.plot(k_list, point_analysis)
plot.xlabel('Number of Clusters')
plot.ylabel('Score')
plot.title('Elbow Curve')
plot.savefig('Elbow.png', dpi=300, bbox_inches='tight')
plot.show()

train_scores_norm, test_scores_norm = standardizer(train_scores, test_scores)
comb_by_average = average(test_scores_norm)
comb_by_maximization = maximization(test_scores_norm)
comb_by_aom = aom(test_scores_norm, 5) # 5 groups
comb_by_moa = moa(test_scores_norm, 5) # 5 groups

evaluate_print('Combination by Average', dy_test, comb_by_average)
evaluate_print('Combination by Maximization', dy_test, comb_by_maximization)
evaluate_print('Combination by AOM', dy_test, comb_by_aom)
evaluate_print('Combination by MOA', dy_test, comb_by_moa)


dy_train_pred = clf_s.labels_ 
dy_train_scores = clf_s.decision_scores_ 

# get the prediction on the test data
dy_test_pred = clf_s.predict(d_test) 
dy_test_scores = clf_s.decision_function(d_test) 


print(dy_train_scores)

from pyod.utils.data import evaluate_print
# evaluate and print the results
print("\nOn Training Data:")
evaluate_print(clf_name, dy_train, dy_train_scores)
print("\nOn Test Data:")
evaluate_print(clf_name, dy_test, dy_test_scores)

visualize(clf_name, d_train, dy_train, d_test, dy_test, dy_train_pred,
              dy_test_pred, show_figure=True, save_figure=True)
