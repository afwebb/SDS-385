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
        X_train = np.concatenate(X_split[:i] + X_split[(i+1):])
        y_train = np.concatenate(y_split[:i] + y_split[(i+1):])
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
    while not test_converge(err, 10e-6):
        u = b + 0.1*np.dot(X.T, y[:,0]-np.dot(X,b))/(2*y.shape[0])
        b = calc_prox(u, alpha)
        err.append(calc_mse(np.dot(X, b), y[:,0])+alpha*sum(abs(b)))
    return b, err

#Run proximal descent with momentum
def run_mom(X, y, alpha):
    b = None
    b = np.random.rand(X.shape[1])
    err = []
    s = [1,]
    b_vec = [b.copy(),]
    z = b.copy()

    while not test_converge(err, 10e-6):
        print calc_mse(np.dot(X, b), y[:,0])
        u = z + 0.1*np.dot(X.T, y[:,0]-np.dot(X,z))/(2*y.shape[0])
        b = calc_prox(u, alpha)
        b_vec.append(b)
        s.append((1+np.sqrt(1+4*s[-1]**2))/2)
        z = b_vec[-1] + ((s[-1]-1)/s[-2])*(b_vec[-1]-b_vec[-2])
        err.append(calc_mse(np.dot(X, b), y[:,0])+alpha*sum(abs(b)))
    return b, err

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
mse_proximal=[]
mse_cp=[]
mse_mom=[]
vec_lambda=np.logspace(-1.3, 1, 20)
vec_lambda=vec_lambda[1:]
vec_coef=[]
vec_coef_prox=[]
vec_coef_mom=[]

#Iterate over values of lambda
for alpha in vec_lambda:
    print alpha

    #Perform the lasso fit over whole sample
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    y_pred = lasso.predict(X)
    mse_train.append(calc_mse(y, y_pred))

    #Perform lasso fit using cross-validation
    mse_test_5.append(cross_val(X, y, alpha, 5))

    #Calc mallow's cp
    mse_cp.append(calc_cp(y,y_pred, lasso.coef_))

    #Perform proximal gradient method
    b_prox, err_prox = run_prox(X, y, alpha)
    y_prox = np.dot(X, b_prox)
    mse_proximal.append(calc_mse(y, y_prox))

    #Perform proximal gradient with momentum
    b_mom, err_mom = run_mom(X, y, alpha)
    y_mom = np.dot(X, b_mom)
    mse_mom.append(calc_mse(y, y_mom))

    #Add coefficients
    vec_coef.append(lasso.coef_)
    vec_coef_prox.append(b_prox)
    vec_coef_mom.append(b_mom)


#Plot mse
plt.figure(2)
plt.plot(np.log10(vec_lambda), mse_train, label='Training Data')
plt.plot(np.log10(vec_lambda), mse_test_5, label='Test Data 5')
plt.plot(np.log10(vec_lambda), mse_cp, label='Cp')
plt.plot(np.log10(vec_lambda), mse_proximal, label='Proximal')
plt.plot(np.log10(vec_lambda), mse_mom, label='Proximal')
plt.ylabel('MSE')
plt.xlabel('log(lambda)')
plt.legend(loc='upper left')
plt.savefig('result_mse.png', format='png')

#Plot coefficient values from scikitlearn as a functino of lambda
plt.figure(3)
vec_coef = np.asarray(vec_coef).T
for i in xrange(vec_coef.shape[0]):
    plt.plot(np.log10(vec_lambda), vec_coef[i,:])
    plt.ylabel('Coefficient Value')
    plt.xlabel('log lambda')
    plt.savefig('coef_lasso.png',format='png')

#Plot coefficient values from prox as a functino of lambda
plt.figure(4)
vec_coef_prox = np.asarray(vec_coef_prox).T
for i in xrange(vec_coef_prox.shape[0]):
    plt.plot(np.log10(vec_lambda), vec_coef_prox[i,:])
    plt.ylabel('Coefficient Value')
    plt.xlabel('log lambda')
    plt.savefig('coef_prox.png',format='png')


#Plot coefficient values from momentum as a functino of lambda
plt.figure(5)
vec_coef_mom = np.asarray(vec_coef_mom).T
for i in xrange(vec_coef_mom.shape[0]):
    plt.plot(np.log10(vec_lambda), vec_coef_mom[i,:])
    plt.ylabel('Coefficient Value')
    plt.xlabel('log lambda')
    plt.savefig('coef_mom.png',format='png')

#Plot the loss for the optimal lambda
#Get min lambda, run descent, plot likliehood
b_prox, err_prox = run_prox(X, y, 0.1)
b_mom, err_mom = run_mom(X, y, 0.1)

plt.figure(6)
plt.plot(err_prox, label='Proximal')
plt.plot(err_mom, label='With Momentum')
plt.ylabel('L')
plt.xlabel('iteration')
plt.legend(loc='upper right')
plt.savefig('likeliehood.png',format='png')
