# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 16:52:23 2022

@author: ChenMingfeng
"""

import numpy as np
import random
from data import test_data, draw_ecg
import scipy.linalg as lp
import scipy.signal
from omp import *
import matplotlib.pyplot as plt
import bsbl

n_rows = 160  # row number of the dictionary matrix
n_cols = 1280  # column number
blkNum = 40  # nonzero block number
blkLen = 32  # block length
SNR = 20  # Signal-to-noise ratio
iterNum = 1  # number of experiments (100)
r = 0.95  # intra-correlation
ecg = test_data().reshape(112, 1280)

# generate nonzero block coefficients
blks = np.zeros((blkNum, blkLen))
blks[:, 0] = np.random.randn(blkNum)
for i in range(1, blkLen):
    blks[:, i] = r * blks[:, i - 1] + np.sqrt(1.0 - r**2) * np.random.randn(blkNum)
    
# ===========================================================================
# put blocks at random locations and align with block partition (no overlap)
# ===========================================================================
blk_start_loc = np.arange(0, n_cols, blkLen)
nblock = blk_start_loc.shape[0]
ind_block = np.random.permutation(nblock)
block_slice = [blk_start_loc[i] + np.arange(blkLen) for i in range(nblock)]
#
x = np.zeros(n_cols, dtype="float")
for i in range(blkNum):
    x[block_slice[ind_block[i]]] = blks[i, :]
  
clf = bsbl.bo(
    learn_lambda=1,
    learn_type=1,
    lambda_init=1e-3,
    epsilon=1e-5,
    max_iters=100,
    verbose=1,
)
file = open("bp.txt",'w')
sum=0

for i in range(112):
    #print(i)
    if i !=0:
        continue
    original = ecg[i,:]
    x_i = scipy.fft.dct(original, type=2)
    #draw_ecg(x_i, 'x_i')
    A = define_A(n_rows, n_cols)
    y = np.dot(A, x_i)
    #draw_ecg(y, 'y')
    #x_est,Lambdas = OMP(A, y, 1280)
    #x_est = clf.fit_transform(A, y, blk_start_loc)
    x_est = np.linalg.pinv(A) @ y
    recon = scipy.fft.idct(x_est, type=2)
    prd = np.linalg.norm(original - recon) / np.linalg.norm(original)
    #file.write(str(prd)+"\n")
    sum += prd
    if i==0:
        for j in range(1280):
            file.write(str(recon[j])+"\n")
            #print(recon[j])
    #print("prd is :", prd)
file.close()
avg = sum / 112
#print(avg)
'''
sum = 0
file = open("cr=50% cosamp-test.txt",'w')
for i in range(112):
    print(i)
    original = x[i,:]
    x_i = scipy.fft.dct(x[i,:], type=2)
    A = define_A(n_rows, n_cols)
    #A = np.random.normal(0, 1, [n_rows, n_cols])
    y = np.dot(A, x_i)
    x_est = cosamp(A, y, 256)
    recon = scipy.fft.idct(x_est, type=2)
    prd = np.linalg.norm(original - recon) / np.linalg.norm(original)
    file.write(str(prd)+"\n")
    sum += prd
    print("prd is:", prd)
file.close()
avg = sum / 112
print(avg)

A = np.random.normal(0, 1, [n_rows, n_cols])
# Generate sparse x and noise
x = np.zeros(n_cols)
x[np.random.randint(1, n_cols, [sparsity])] = np.random.chisquare(15, [sparsity])
noise = np.random.normal(0, 1, [n_cols])

u = x + noise

y = np.dot(A, u)

x_est = cosamp(A, y, 20)
# Score estimation
print(np.linalg.norm(x - x_est) / np.linalg.norm(x))
'''