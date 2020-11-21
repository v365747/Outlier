
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
training_data = "train4.txt"
test_data = "test4.txt"
#training_data = "train.txt"
#test_data = "test.txt"
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

"""
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
#
# DMAX
# This is to label data as outlier for training... 
#
DMAX_OUTLIER_50k = 24299489320
#DMAX_OUTLIER_1M = 34299489320

for i in range(len(Mem)):
    #if Mem[i] > DMAX_OUTLIER_1M:
    if Mem_t[i] > DMAX_OUTLIER_50k:
        dy_train.append(1.)
    else:
        dy_train.append(0.)


for i in range(len(Mem_t)):
    #if Mem_t[i] > DMAX_OUTLIER_1M:
    if Mem_t[i] > DMAX_OUTLIER_50k:
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
"""
##################
# LSTM - PyTorch
# https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
# Used above link to learn LSTM-RNN using pyTorch
##################
import torch
import torch.nn as nn
import pandas as pd


#
# Plotting Memory Access (JVM Running in 2 states, Steady state (Train), Terminating (Test)
#
fig_size = plot.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 5
fig = plot.figure()
ax = fig.add_subplot(2,1,1)

plot.rcParams["figure.figsize"] = fig_size
plot.title('Memory Accesses')
plot.ylabel('Normalized Memory range')
plot.xlabel('Time')
plot.grid(True)
plot.autoscale(axis='x', tight=True)
line, = ax.plot(Mem, color='blue', lw = 2)
line1, = ax.plot(Mem_t, color='orange', lw =2)
#line, = ax.plot(feature_2_train, color='blue', lw = 2)
#line1, = ax.plot(feature_2_test, color='orange', lw =2)
ax.set_yscale('log')
plot.title('Time Series Memory Access Curve')
plot.savefig('TS_Memory_train.png', dpi=300, bbox_inches='tight')
plot.show()

#
# Plotting Code Segment:IP JVM {Interpreter + Hot Code}
#

fig_size = plot.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 5
fig = plot.figure()
ax = fig.add_subplot(2,1,1)

plot.rcParams["figure.figsize"] = fig_size
plot.title('Code Segment ')
plot.ylabel('Normalized CS:IP range')
plot.xlabel('Time')
plot.grid(True)
plot.autoscale(axis='x', tight=True)
line, = ax.plot(CS, color='blue', lw = 2)
line1, = ax.plot(CS_t, color='orange', lw =2)
ax.set_yscale('log')
plot.title('Time Series Code Segment:IP Curve')
plot.savefig('TS_CS_IP.png', dpi=300, bbox_inches='tight')
plot.show()



##
# Data Preprocessing
##
# Create numpy array of data.
train_data = np.array(Mem)
test_data = np.array(Mem_t)

# Convert to float
train_data_f = train_data.astype(np.float)
test_data_f = test_data.astype(np.float)

print(train_data_f)

# Since there is quite a bit of variation in memory accesses
# Let us do min/max scaling

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_f_normalized = scaler.fit_transform(train_data_f.reshape(-1, 1))

# No need to implement normalization on test data.

# Data validation 
#print(train_data_f_normalized[:5], train_data_f_normalized[-5:])

# Convert data to tensors.
# PyTorch uses Tensors... (same as numpy array, only difference is Tensor can run on CPU or GPU)
train_data_f_normalized = torch.FloatTensor(train_data_f_normalized).view(-1)

# We need to setup a sequence window for training.
# Starting with 250
LSTM_training_window = 250

# Now for PyTorch, we will label in put data in sequence of training window and + 1.
def sequencer(data, window):
    if data.size == 0 :
        print('Fix data set')
        return []
    input_seq = []
    Length = len(data)
    for i in range(Length - window):
        t_seq = data[i:i+window]
        t_label = data[i+window:i+window+1]
        input_seq.append((t_seq, t_label))
    return input_seq

# Create Sequence
train_IO = sequencer(train_data_f_normalized, LSTM_training_window)

# Data Validation
#print(train_IO[:3])

#### Data Preprocessing is complete...
#### Create LSTM Model

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)


# Now let us train for epochs and test

epochs = 3
for i in range(epochs):
    for seq, labels in train_IO:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1,1, model.hidden_layer_size),
                             torch.zeros(1,1,model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%30 == 1:
        print(f'epoch: {1:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {1:3} loss: {single_loss.item():10.10f}')

# Making next 250 predictions.

future_prediction_window = 250

predicted_test_inputs = train_data_f_normalized[-LSTM_training_window:].tolist()
#print(predicted_test_inputs)

model.eval()

for i in range(future_prediction_window):
    seq = torch.FloatTensor(predicted_test_inputs[-LSTM_training_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        predicted_test_inputs.append(model(seq).item())

# Normalized predicted values.
#print(predicted_test_inputs[future_prediction_window:])

# Do an inverse transform and plot
predicted_memory_accesses = scaler.inverse_transform(np.array(predicted_test_inputs[LSTM_training_window:]).reshape(-1,1))
#print(predicted_memory_accesses)

x = np.arange(1750,2000,1)
#print(x)

#
# Plotting Memory Access (JVM Running in 2 states, Steady state (Train), Terminating (Test)
#
fig_size = plot.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 5
fig = plot.figure()
ax = fig.add_subplot(2,1,1)
 
plot.rcParams["figure.figsize"] = fig_size
plot.title('Memory Accesses LSTM-RNN prediction')
plot.ylabel('Normalized Memory range')
plot.xlabel('Time')
plot.grid(True)
plot.autoscale(axis='x', tight=True)
line, = ax.plot(Mem, color='blue', lw = 2)
line1, = ax.plot(Mem_t, color='orange', lw =2)
line2, = ax.plot(x, predicted_memory_accesses, color='green', lw =2)
ax.set_yscale('log')
plot.title('Time Series Memory Access Curve')
plot.savefig('TS_Memory_train_LSTM.png', dpi=300, bbox_inches='tight')
plot.show()
