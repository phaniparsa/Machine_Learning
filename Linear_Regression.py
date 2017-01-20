# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 22:28:21 2017

@author: useradmin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_frame = pd.read_csv("C:\Users\useradmin\Desktop\Learning\home_data.csv")
data_frame = data_frame[["sqft_living", "price"]]
#print data_frame.head()

temp = np.random.randn(len(data_frame)) < 0.8

train_sample = data_frame[temp]
test_sample = data_frame[~temp]

theta = np.array([0, 0], dtype='float16')

train_sample["x_zero"]=1

alpha = 0.01

cost = np.zeros(15000)

train_sample["sqft_living"] = train_sample["sqft_living"]/train_sample["sqft_living"].max()
train_sample["price"] = train_sample["price"]/train_sample["price"].max()

print train_sample.head()
for i in range(15000):
    #print "here"
    temp = np.dot(train_sample[["x_zero", "sqft_living"]], np.transpose(theta))-train_sample["price"]
    theta[0] = theta[0] - alpha*(np.sum(np.multiply(temp, train_sample["x_zero"]))/len(train_sample))
    theta[1] = theta[1] - alpha*(np.sum(np.multiply(temp, train_sample["sqft_living"]))/len(train_sample))
    
    
    cost[i] = np.sum(np.dot(train_sample[["x_zero", "sqft_living"]], np.transpose(theta))-train_sample["price"])/(2*len(train_sample))

test_sample["sqft_living"] = test_sample["sqft_living"]/test_sample["sqft_living"].max()
test_sample["price"] = test_sample["price"]/test_sample["price"].max()
test_sample["predicted_price"] = theta[0] + theta[1]*test_sample["sqft_living"]

#np.dot(test_sample[["x_zero", "sqft_living"]], np.transpose(theta))-train_sample["price"]

#plt.scatter([i for i in range(1500)], cost)
print theta
plt.plot(test_sample['sqft_living'],test_sample['price'],'.',
        test_sample['sqft_living'],test_sample["predicted_price"],'-')
