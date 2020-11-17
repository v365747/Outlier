
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

contamination = 0.1
n_train = 200
n_test = 100

X_train, y_train, X_test, y_test = generate_data(n_train=n_train, n_test=n_test, contamination=contamination)

n_clf = 20
k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]

# -------------------------------------------
# Preprocess train file for python processing by
# tr '\r' '\n' < File > File_1
# -------------------------------------------
CS = []
Mem = []
with open("DLA_do_object_1.csv", "r") as file:
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
with open("DLA_OBJ_ITER_1.csv", "r") as file:
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
plot.show()
exit()

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
for i in range(len(Mem)):
    if Mem[i] > 34299489320:
        dy_train.append(1.)
    else:
        dy_train.append(0.)

#dy_train_f = np.zeros(len(d_train) - 570)
#dy_train_t = np.ones(570)
#dy_train = np.r_[dy_train_f, dy_train_t]

for i in range(len(Mem_t)):
    if Mem_t[i] > 34299489320:
        dy_test.append(1.)
    else:
        dy_test.append(0.)

#dy_test_t = np.ones(len(d_test) - 570)
#dy_test_f = np.zeros(570)
#dy_test = np.r_[dy_test_t, dy_test_f]
#print(y_test)
#print(d_train)


print('Combining {n_clf} kNN detectors'.format(n_clf=n_clf))

for i in range(n_clf):
    k = k_list[i]
    clf_name = 'KNN'
    clf = KNN(n_neighbors=k, method='largest')
    #clf.fit(X_train)
    clf.fit(d_train_norm)
    train_scores[:, i] = clf.decision_scores_
    test_scores[:, i] = clf.decision_function(d_test_norm)

train_scores_norm, test_scores_norm = standardizer(train_scores, test_scores)
comb_by_average = average(test_scores_norm)
comb_by_maximization = maximization(test_scores_norm)
comb_by_aom = aom(test_scores_norm, 5) # 5 groups
comb_by_moa = moa(test_scores_norm, 5) # 5 groups

evaluate_print('Combination by Average', dy_test, comb_by_average)
evaluate_print('Combination by Maximization', dy_test, comb_by_maximization)
evaluate_print('Combination by AOM', dy_test, comb_by_aom)
evaluate_print('Combination by MOA', dy_test, comb_by_moa)

# get the prediction labels and outlier scores of the training data
#y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
#y_train_scores = clf.decision_scores_  # raw outlier scores
dy_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
dy_train_scores = clf.decision_scores_  # raw outlier scores

# get the prediction on the test data
#y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
dy_test_pred = clf.predict(d_test)  # outlier labels (0 or 1)
#y_test_scores = clf.decision_function(X_test)  # outlier scores
dy_test_scores = clf.decision_function(d_test)  # outlier scores


from pyod.utils.data import evaluate_print
# evaluate and print the results
print("\nOn Training Data:")
#evaluate_print(clf_name, y_train, y_train_scores)
evaluate_print(clf_name, dy_train, dy_train_scores)
print("\nOn Test Data:")
#evaluate_print(clf_name, y_test, y_test_scores)
evaluate_print(clf_name, dy_test, dy_test_scores)

#visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
#              y_test_pred, show_figure=True, save_figure=True)
visualize(clf_name, d_train, dy_train, d_test, dy_test, dy_train_pred,
              dy_test_pred, show_figure=True, save_figure=True)
