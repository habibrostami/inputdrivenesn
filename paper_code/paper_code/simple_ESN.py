import numpy as np
from scipy import linalg
import networkx as nx
from tqdm import tqdm
import csv  # Importing csv module to save data
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import torch
from torch import nn
import torch.optim as optim
import os
import time
start_time = time.time()
random_seed = 42
# Set random seeds for reproducibility
np.random.seed(random_seed)  # Set seed for NumPy
torch.manual_seed(random_seed)  # Set seed for PyTorch
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)  # Set seed for all GPUs (if applicable)

# Generate MG time series
def generate_multi_dimensional_MG_sequence(sequenceLength, tau, dimensions):
    samplelength = sequenceLength
    initWashoutLength = sequenceLength // 10

    inputSequence = np.ones((sequenceLength, dimensions))
    outputSequence = np.ones((sequenceLength, dimensions))

    incrementsperUnit = 10
    genHistoryLength = tau * incrementsperUnit
    seed = 1.2 * np.ones((int(genHistoryLength), dimensions)) + 0.2 * (np.random.rand(int(genHistoryLength), dimensions) - 0.5)
    oldval = 1.2 * np.ones((1, dimensions))
    genHistory = seed
    speedup = 1

    sample = np.zeros((samplelength, dimensions))
    step = 0

    for n in range(1, samplelength + initWashoutLength + 1):
        for i in range(incrementsperUnit * speedup):
            step += 1
            tauval = genHistory[int(step % genHistoryLength)]
            newval = oldval + (0.2 * tauval / (1.0 + tauval ** 10) - 0.1 * oldval) / incrementsperUnit
            genHistory[int(step % genHistoryLength)] = oldval
            oldval = newval

        if n > initWashoutLength:
            sample[n - initWashoutLength - 1] = newval

    testseq = sample - 1.0
    outputSequence = testseq
    inputSequence[1:] = outputSequence[:-1]
    inputSequence[0] = 0.0

    return inputSequence, outputSequence

# Example usage:
sequenceLength = 20000
tau = 29
dimensions = 1
inputSequence, outputSequence = generate_multi_dimensional_MG_sequence(sequenceLength, tau, dimensions)
print("Input Sequence Shape:", inputSequence.shape)
print("Output Sequence Shape:", outputSequence.shape)


print(f'W = {W}')

# -*- coding: utf-8 -*-
"""
A minimalistic Echo State Networks demo with Mackey-Glass (delay 17) data
in "plain" scientific Python.
from https://mantas.info/code/simple_esn/
(c) 2012-2020 Mantas LukoÅ¡eviÄius
Distributed under MIT license https://opensource.org/licenses/MIT
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.sparse import random


# numpy.linalg is also an option for even fewer dependencies
data = outputSequence.squeeze()
# load the data
pre_train = 0.2
pre_test = 0.8
len_of_data = len(data)
trainLen = int(pre_train * len_of_data)
testLen = len_of_data - trainLen
initLen = 100




# plot some of it
# plt.figure(10).clear()
# plt.plot(data[:1000])
# plt.title('A sample of data')

# generate the ESN reservoir
inSize = outSize = 1
resSize = 50

a = 0.3  # leaking rate
np.random.seed(42)
Win = (np.random.rand(resSize, 1 + inSize) - 0.5) * 1
print(f'Win = {Win}')
# W = (np.random.rand(resSize, resSize) - 0.5)
W = (np.random.rand(resSize, resSize) -0.5)
# density = 0.2  # Sparsity level

# Generate a random sparse matrix with given density
# W = (np.random.rand(resSize, resSize) < density) * np.random.rand(resSize, resSize)
# W = W.astype(np.float64)



# threshold = 0
# W[W < threshold] = 0


# normalizing and setting spectral radius (correct, slow):
rhoW = max(abs(linalg.eig(W)[0]))
W *= 1.25 / rhoW
# print(f'W[3] : {W[3]}')
# allocated memory for the design (collected states) matrix
X = np.zeros((1 + inSize + resSize, trainLen - initLen))

# set the corresponding target matrix directly
Yt = data[None, initLen + 1:trainLen + 1]

# run the reservoir with the data and collect X
x = np.zeros((resSize, 1))
for t in range(trainLen):
    u = data[t]
    x = (1 - a) * x + a * np.tanh(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))
    if t >= initLen:
        X[:, t - initLen] = np.vstack((1, u, x))[:, 0]
print(" X shape : ", np.array(X).shape)
# train the output by ridge regression
reg = 4.840e-07  # regularization coefficient
# using scipy.linalg.solve:
A = np.dot(X, X.T) + reg * np.eye(1 + inSize + resSize)
B = np.dot(X, Yt.T)
Wout = np.linalg.solve(A, B).T
Y = np.zeros((outSize, testLen))
u = data[trainLen]
# calculate ouput
for t in range(testLen):
    x = (1 - a) * x + a * np.tanh(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))
    y = np.dot(Wout, np.vstack((1, u, x)))
    Y[:, t] = y
    u = y

errorLen = 500
mse = sum(np.square(data[trainLen + 1:trainLen + errorLen + 1] - Y[0, 0:errorLen])) / errorLen


data_tensor = torch.tensor(data)
# print(f'prediction[:, 0:errorLen] : {prediction[0:errorLen].shape}')
# print((f'data_tensor[trainLen + 1: trainLen + errorLen + 1, :] : {data_tensor[trainLen + 1: trainLen + errorLen + 1, :].shape}'))
difference = data[trainLen + 1:trainLen + errorLen + 1] - Y[0, 0:errorLen]
squared_difference = difference ** 2
mse = np.mean(squared_difference)
# print(f'mse : {mse}')
rmse = np.sqrt(mse)
range_ = (np.max(data[trainLen + 1:trainLen + errorLen + 1]) - np.min(data[trainLen + 1:trainLen + errorLen + 1]))
nrmse_range = rmse / range_
print(f'nrmse test : {nrmse_range}')
print(f'mse test : {mse}')