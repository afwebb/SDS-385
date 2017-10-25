import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import datasets
from sklearn.linear_model import Lasso, lasso_path

#Calculate mse of estimate
def calc_mse(y_pred, y):
    mse = 0
    for i,j in zip(y,y_pred):
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
def calc_cp(y_pred, y, coef):
    cp = calc_mse(y_pred, y) 
    sl = np.count_nonzero(coef)
    cp += 2*sl*cp/(len(y)-sl)
    return cp

#Peform cross validation, splitting the sample into n parts. Return mse
def cross_val(X, y, alpha, n):
    X_split = np.array_split(X, n)
    y_split = np.array_split(y, n)
    err = 0
    lasso = Lasso(alpha=alpha)

    for i in xrange(n):
        X_train = np.concatenate(X_split[:i] + X_split[(i+1):])
        y_train = np.concatenate(y_split[:i] + y_split[(i+1):])
        X_test, y_test = X_split[i], y_split[i]
        lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_test)
        err+=sk.metrics.mean_squared_error(y_pred, y_test)
        
    return err/n


#Read in data
X = pd.read_csv('diabetesX.csv')
col_names=X.columns.values.tolist()
y = pd.read_csv('diabetesY.csv', header=None)
y = y.values
X = X.values
X,y = sk.utils.shuffle(X,y)
X /= X.std(axis=0)
n_samples = X.shape[0]

#Calculate mse for test data, train data, and Cp estimate. Calc coefficients for all lambda
plt.figure(1)
mse_train=[]
mse_test_5=[]
mse_test_10=[]
mse_cp=[]
vec_lambda=np.linspace(0,19,200)
vec_lambda=vec_lambda[1:]
vec_coef=[]

#Iterate over values of lambda
for alpha in vec_lambda:
    #Perform the lasso fit over whole sample
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    y_pred = lasso.predict(X)
    mse_train.append(calc_mse(y, y_pred))

    #Perform lasso fit using cross-validation
    mse_test_5.append(cross_val(X, y, alpha, 5))
    mse_test_10.append(cross_val(X, y, alpha, 10))

    #Calc mallow's cp
    mse_cp.append(calc_cp(y,y_pred, lasso.coef_))

    #Plot coefficients
    vec_coef.append(lasso.coef_)
    if alpha%2==0 and alpha!=0:
        plt.plot(lasso.coef_, label="Lambda: "+str(alpha))

#Plot coefficients values for various lambdas
plt.ylabel('Lasso Coefficients')
plt.legend(loc='upper right')
plt.savefig('result_lasso_coef_test.png', format='png')

#Plot mse
plt.figure(2)
plt.plot(np.log10(vec_lambda), mse_train, label='In sample')
plt.plot(np.log10(vec_lambda), mse_test_5, label='5 fold CV')
plt.plot(np.log10(vec_lambda), mse_test_10, label='10 fold CV')
plt.plot(np.log10(vec_lambda), mse_cp, label='Cp')
plt.ylabel('MSE')
plt.xlabel('log(lambda)')
plt.legend(loc='upper left')
plt.savefig('result_lasso_mse.png', format='png')

#Plot coefficient values as a functino of lambda
plt.figure(3)
vec_coef = np.asarray(vec_coef).T
for i in xrange(len(vec_coef[:,0])):
    plt.plot(np.log10(vec_lambda), vec_coef[i,:])#, label=col_names[i])
    plt.ylabel('Coefficient Value')
    plt.xlabel('log lambda')
    plt.savefig('coef_lambda.png',format='png')
