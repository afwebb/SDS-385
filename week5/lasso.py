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

#Calculate the cp statistic
def calc_cp(y_pred, y, X):
    mse = calc_mse(y_pred, y, X) 
    sl = len([x for x in y_pred if x > 0])
    sigma = mse
    if len(y)!=sl:
        sigma = mse/(len(y)-sl)
    cp = mse + 2*sl*sigma/len(y)
    return cp

#Read in data
X = pd.read_csv('diabetesX.csv')
col_names=X.columns.values.tolist()
y = pd.read_csv('diabetesY.csv')
y = y.values
X = X.values

#Use the rest for X. Scale X by the size of the vector
X /= X.std(axis=0)
n_samples = X.shape[0]
X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]

#Calculate mse for test data, train data, and Cp estimate. Calc coefficients for all lambda
plt.figure(1)
mse_test=[]
mse_train=[]
mse_cp=[]
vec_lambda=np.linspace(0,9,100)
vec_coef=[]
for alpha in vec_lambda:
    lasso = Lasso(alpha=alpha)
    y_pred = lasso.fit(X_train, y_train).predict(X_test)
    mse_test.append(calc_mse(y_test,y_pred,X_test))
    mse_train.append(calc_mse(y_train,y_pred,X_train))
    mse_cp.append(calc_cp(y_train,y_pred,X_train))
    vec_coef.append(lasso.coef_)
    if alpha%2==0 and alpha!=0:
        plt.plot(lasso.coef_, label="Lambda: "+str(alpha))

#Plot coefficients values for various lambdas
plt.ylabel('Lasso Coefficients')
plt.legend(loc='upper right')
plt.savefig('result_lasso_coef.png', format='png')

#Plot mse
plt.figure(2)
plt.plot(vec_lambda, mse_train, label='Training Data')
plt.plot(vec_lambda, mse_test, label='Test Data')
plt.plot(vec_lambda, mse_cp, label='Cp')
plt.ylabel('MSE')
plt.xlabel('lambda')
plt.legend(loc='upper right')
plt.savefig('result_lasso_mse.png', format='png')

#Plot coefficient values as a functino of lambda
plt.figure(3)
vec_coef = np.asarray(vec_coef).T
for i in xrange(10):#len(vec_coef[:,0])):
    print i
    plt.plot(vec_lambda[:50], vec_coef[i,:50], label=col_names[i])
    plt.ylabel('Coefficient Value')
    plt.xlabel('lambda')
    plt.legend()
    plt.savefig('coef_lambda.png',format='png')