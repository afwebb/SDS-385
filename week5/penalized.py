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

#calculate prediction vector, s(y)
def calc_s(y, lamb):
    s = abs(y)-lamb
    s[s<0]=0
    s = np.sign(y)*s
    return s

#Plot of likliehood function as a function of theta
plt.figure(3)
#Plot of min likliehood, s(y) as a function of y
plt.figure(4)
for lam in [0, 2, 4, 6, 8]:
    y_vec = np.linspace(-10,10,100)
    s_vec = calc_s(y_vec, lam)
    plt.figure(3)
    for y in [4, 8]:
        if lam%8==0:
            likliehood=0.5*(y-y_vec)*(y-y_vec)+lam*abs(y_vec)
            plt.plot(y_vec, likliehood, label='lambda='+str(lam)+', y='+str(y))
    plt.figure(4)
    plt.plot(y_vec, s_vec, label='lambda='+str(lam))

plt.figure(3)
plt.ylabel('S(y)')
plt.xlabel('theta')
plt.legend(loc='upper right')
plt.savefig('theta_s.png', format='png')

plt.figure(4)
plt.ylabel('S(y)')
plt.xlabel('y')
plt.legend(loc='lower right')
plt.savefig('theta_y.png', format='png')

#Define vector length, and sigma vector
n=10000
sigma = np.random.random(n)
y = np.random.random(n)
plt.figure(1)
#Calculate MSE as a function of lambda for various levels of sparsity
for z in [0.9, 0.5, 0.25, 0.1, 0.001]:
    theta = scipy.sparse.random(n, 1, density=z)#np.random.choice([0, 1], size=(n), p=[1-z, z])
    theta=theta.A
    norm = np.exp(-np.square(y-theta)/(2*np.square(sigma)))
    mse=[]
    vec_lambda=np.linspace(0,1.5,100)
    for lamb in vec_lambda:
        s = calc_s(y, lamb)
        mse.append(calc_mse(s, theta))

    plt.plot(vec_lambda, mse, label='sparsity='+str(z))

plt.ylabel('MSE')
plt.xlabel('lambda')
plt.legend(loc='upper right')
plt.savefig('result_pen.png', format='png')

#Plot S(y) as a function of input theta
plt.figure(2)
for lam in [0, 4, 8]:
    #Define theta vector, calculate s(y)
    th = scipy.sparse.random(n, 1, density=0.001)#np.random.choice([0, 1], size=(n), p=[1-0.001, 0.001])
    th = th.A
    s = calc_s(y, lam)

    # Calculate the point density
    #xy = np.vstack([th,s])
    z = scipy.stats.gaussian_kde(th+s)(th+s)

    plt.scatter(th, s, label='lambda='+str(lam), c=z, s=100, edgecolor='')

plt.xlabel('theta')
plt.ylabel('s')
plt.legend(loc='upper right')
plt.savefig('result_theta.png', format='png')
    
        
        
