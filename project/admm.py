import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import datasets
from sklearn.linear_model import Lasso, lasso_path
from time import time

#Calculate mse of estimate
def calc_mse(y_pred, y):
    mse = 0
    for i,j in zip(y,y_pred):
        mse+=np.square(i-j)
    mse=mse/len(y)
    return mse

#calculate prediction vector
def calc_prox(u, lamb):
    prox = abs(u)-lamb
    prox[prox<0]=0
    prox = np.sign(u)*prox
    return prox

#Calculate the cp statistic
def calc_cp(y_pred, y, coef):
    cp = calc_mse(y_pred, y) 
    sl = np.count_nonzero(coef)
    cp += 2*sl*cp/(len(y)-sl)
    return cp

#Check the convergance of the log likliehood function within some criteria, e. Return true converged
def test_converge(l, e):
    if len(l)>5:
        conv = (l[-1]-l[-2])/l[-1]
        if abs(conv) < e:
            return 1
        else:
            return 0
    else:
        return 0

#Peform cross validation, splitting the sample into n parts. Return mse
def cross_val(X, y, alpha, n):
    X_split = np.array_split(X, n)
    y_split = np.array_split(y, n)
    err = 0
    lasso = Lasso(alpha=alpha)

    for i in xrange(n):
        X_train = scipy.vstack([ X_split[:i] , X_split[(i+1):] ] )
        y_train = scipy.vstack( [ y_split[:i] + y_split[(i+1):] ] )
        X_test, y_test = X_split[i], y_split[i]
        lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_test)
        err+=sk.metrics.mean_squared_error(y_pred, y_test)
        
    return err/n

#Run proximal gradient descent
def run_prox(X, y, alpha):
    b = None
    b = np.random.rand(X.shape[1])
    err = []
    while not test_converge(err, 10e-5):
        u = b + 0.1 * (X.transpose()).dot( y - X.dot(b) )/(2*y.shape[0]) + alpha * (b > 0).astype(float) - alpha * (b < 0).astype(float)
        b = calc_prox(u, alpha)
        err.append(calc_mse(X.dot(b), y )+alpha*sum(abs(b)))
    return b, err

#Run admm
def run_admm(X, y, alpha, rho = 1):

    b = None
    b = np.random.rand(X.shape[1])
    err = []
    z = np.copy(b)
    u=np.zeros(X.shape[1])

    x_inv = np.dot( X.transpose(), X ) + rho * np.identity( X.shape[1] )
    xy = np.dot(X.transpose(), y)

    while not test_converge(err, 10e-5):
        print calc_mse(np.dot(X, b), y)+alpha*sum(abs(z))
        b = np.linalg.solve(x_inv, xy + rho* (z - u ))
        z = calc_prox( b + u , alpha/rho)
        u = u + b - z
        err.append(calc_mse(np.dot(X, b), y)+alpha*sum(abs(z)))
    return b, err

#Run proximal gradient descent with momentum
def run_mom(X, y, alpha):
    b = None
    b = np.random.rand(X.shape[1])
    err = []
    s = [1,]
    b_vec = [b.copy(),]
    z = b.copy()

    while not test_converge(err, 10e-8):
        u = z + 0.1*np.dot(X.transpose(), y - X.dot(z))/(2*y.shape[0])
        b = calc_prox(u, alpha)
        b_vec.append(b)
        s.append((1+np.sqrt(1+4*s[-1]**2))/2)
        z = b_vec[-1] + ((s[-1]-1)/s[-2])*(b_vec[-1]-b_vec[-2]) + alpha * (b > 0).astype(float) - alpha * (b < 0).astype(float)
        err.append(calc_mse( X.dot(b), y )+alpha*sum(abs(b)))
    return b, err

#Read in data
X,y = sk.datasets.load_svmlight_file('test_data')# offset=i*batch_size, length=batch_size)
#col_names=X.columns.values.tolist()
#y = y.values
#X = X.values
#X,y = sk.utils.shuffle(X,y)
n_samples = X.shape[0]

#Plot the loss for proximal, momentum, and admm
time0 = time()
b_prox, err_prox = run_prox(X, y, 0.01)
time1 = time()
print "past prox"

b_mom, err_mom = run_mom(X, y, 0.01)
time2 = time()
print "past mom"

b_admm, err_admm = run_admm(X, y, 0.001)
time3 = time()
print "admm"

time_prox = time1-time0
time_mom = time2-time1
time_admm = time3-time2

plt.figure(6)
plt.semilogx(err_prox, label='Proximal')
plt.semilogx(err_mom, label='With Momentum')
plt.semilogx(err_admm, label='ADMM')
plt.ylabel('L')
plt.xlabel('iteration')
plt.legend(loc='upper right')
plt.savefig('plots/admm.png',format='png')
