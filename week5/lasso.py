import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import datasets
from sklearn.linear_model import Lasso

#Calculate mse of estimate
def calc_mse(y_pred, y, X):
    mse = 0
    xy = np.dot(X.T, y)
    for i,j in zip(y,xy):
        mse+=np.square(i-j)
    mse=mse/len(y)
    return mse

#calculate prediction vector
def calc_s(y, lamb):
    s = abs(y)-lamb
    s[s<0]=0
    s = np.sign(y)*s
    return s

#Read in data
X = pd.read_csv('diabetesX.csv')
y = pd.read_csv('diabetesY.csv')
y = y.values
X = X.values

#Use the rest for X. Scale X by the size of the vector
X /= X.std(axis=0)
n_samples = X.shape[0]
X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]

plt.figure(1)
mse=[]
vec_lambda=np.linspace(0,9,100)
for alpha in vec_lambda:
    lasso = Lasso(alpha=alpha)
    y_pred = lasso.fit(X_train, y_train).predict(X_test)
    #s = calc_s(y_pred, alpha)
    mse.append(calc_mse(y_test,y_pred,X_test))
    if alpha%2==0 and alpha!=0:
        plt.plot(lasso.coef_, label="Lambda: "+str(alpha))
    
plt.ylabel('Lasso Coefficients')
plt.legend(loc='upper right')
plt.savefig('result_lasso_coef.png', format='png')

plt.figure(2)
plt.plot(vec_lambda, mse)
plt.ylabel('MSE')
plt.xlabel('lambda')
plt.savefig('result_lasso_mse.png', format='png')
