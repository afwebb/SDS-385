import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import datasets
from sklearn import preprocessing

#Calculate mse of estimate
def calc_mse(s, theta):
    mse = 0
    for i,j in zip(theta,s):
        mse+=np.square(i-j)
    mse=mse/len(theta)
    return mse

#calculate prediction vector        
def calc_s(y, lamb):
    s = abs(y)-lamb
    s[s<0]=0
    s = np.sign(y)*s
    return s

#Define vector length, and sigma vector
n=1000
sigma = np.random.random(n)
y = np.random.random(n)
plt.figure()

for z in [0.5, 0.25, 0.1, 0.001]:
    theta = np.random.choice([0, 1], size=(n), p=[1-z, z])
    norm = np.exp(-np.square(y-theta)/(2*np.square(sigma)))
    mse=[]
    vec_lambda=np.linspace(0,1.2,100)
    for lamb in vec_lambda:
        s = calc_s(y, lamb)
        mse.append(calc_mse(s, theta))

    plt.plot(vec_lambda, mse, label='sparsity='+str(z))

plt.ylabel('MSE')
plt.xlabel('lambda')
plt.legend(loc='upper right')
plt.savefig('result_pen.png', format='png')
    
    
        
        
