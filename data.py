# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 20:06:08 2022

@author: ChenMingfeng
"""

import wfdb
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

# 画心电图
def draw_ecg(x,notes):
    plt.figure()
    plt.plot(x)
    plt.title(notes)
    plt.show()
    plt.hist(x)
    plt.title(notes)
    plt.show()
    
# 压缩采样
def sample(signal):    
    N = len(signal)   # 原始信号维度
    M = 48   # 采样数
    samples = random.sample(range(N),M)
    samples.sort()
    Theta = np.zeros((M,N))
    for i,j in enumerate(samples):
      Theta[i,j]=1
    y = np.dot(Theta,signal)
    #draw_ecg(y,"测量信号")
    return y, Theta

# 预处理
def preprocess(y,Theta):
    Theta_T = np.transpose(Theta)
    y_pre = np.dot(Theta_T,y)
    #draw_ecg(y_pre, "升维信号")
    return y_pre 
'''   
def define_A(M,N):
    Define the measurement matrix
    :args
    M: number of row
    N: number of column
    :returns
    A: M x N matrix with normalized columns drawn from N(0,1)
    
    A = []
    for i in range(N):
        col = np.random.randn(M)
        col /= norm2(col)
        
        A.append(col)
    A = np.array(A).T
    return A 
'''
def get_ecg_data(num):
    '''
    读取心电信号文件
    sampfrom: 设置读取心电信号的 起始位置，sampfrom=0表示从0开始读取，默认从0开始
    sampto：设置读取心电信号的 结束位置，sampto = 1500表示从1500出结束，默认读到文件末尾
    channel_names：设置设置读取心电信号名字，必须是列表，channel_names=['MLII']表示读取MLII导联线
    channels：设置读取第几个心电信号，必须是列表，channels=[0, 3]表示读取第0和第3个信号，注意信号数不确定
    :return:
    '''
    #address='../mit-bih-ecg-compression-test-database-1.0.0/'
    address = 'C:/Users/ChenMingfeng/Downloads/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0/'
    record = wfdb.rdrecord(address+num, sampto=1280, channels=[0])
    
    return record.p_signal

def get_train_data(index):
    signal = get_ecg_data(train_nums[index])
    return signal
    #return torch.FloatTensor(signal), torch.FloatTensor(signal)  
        
def get_test_data(index):    
    signal = get_ecg_data(test_nums[index])
    return signal
    #return torch.FloatTensor(signal), torch.FloatTensor(signal)  

def get_csnet_train_data(index):
    signal = get_ecg_data(train_nums[index])
    #y, Theta = sample(signal)
    #y_pre = preprocess(y, Theta)
    y = np.dot(A, signal)
    y_pre = np.dot(A.T, signal)
    
    return torch.FloatTensor(y_pre), torch.FloatTensor(signal)
  
def get_csnet_test_data(index):
    signal = get_ecg_data(test_nums[index])
    #y, Theta = sample(signal)
    #y_pre = preprocess(y, Theta)
    y = np.dot(A, signal)
    y_pre = np.dot(A.T, signal)
    
    return torch.FloatTensor(y_pre), torch.FloatTensor(signal)

class TrainData(torch.utils.data.Dataset):
    def __init__(self):
        super(TrainData, self).__init__()

    def __len__(self):
        return 1008

    def __getitem__(self, index):
        return get_train_data(index)
    
class TestData(torch.utils.data.Dataset):
    def __init__(self):
        super(TestData, self).__init__()

    def __len__(self):
        return 112

    def __getitem__(self, index):
        return get_test_data(index)
    
class CSNet_TrainData(torch.utils.data.Dataset):
    def __init__(self):
        super(CSNet_TrainData, self).__init__()
        
    def __len__(self):
        return 1008
    
    def __getitem__(self, index):
        return get_csnet_train_data(index)
    
class CSNet_TestData(torch.utils.data.Dataset):
    def __init__(self):
        super(CSNet_TestData, self).__init__()
        
    def __len__(self):
        return 112
    
    def __getitem__(self, index):
        return get_csnet_test_data(index)

train_file = open('C:/Users/ChenMingfeng/Downloads/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0/train_file.txt', 'r')
test_file = open('C:/Users/ChenMingfeng/Downloads/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0/test_file.txt', 'r')
train_nums = []
test_nums = []
#A = define_A(48,256)

for line in train_file.readlines():
    data = line.split('\n')
    train_nums.append(data[0])
    
for line in test_file.readlines():
    data = line.split('\n')
    test_nums.append(data[0])
'''    
def test_data():
    data=[]
    for i in range(112):
        data=np.append(data,get_test_data(i))
        
    return data
    
'''
trainloader = torch.utils.data.DataLoader(dataset=TrainData(),
                                          batch_size=32,
                                          shuffle=True)

testloader = torch.utils.data.DataLoader(dataset=TestData(),
                                         batch_size=1,
                                         shuffle=False)

csnet_trainloader = torch.utils.data.DataLoader(dataset=CSNet_TrainData(),
                                                batch_size=32,
                                                shuffle=True)

csnet_testloader = torch.utils.data.DataLoader(dataset=CSNet_TestData(),
                                               batch_size=1,
                                               shuffle=False)
