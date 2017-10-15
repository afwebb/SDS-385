import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import datasets
from sklearn import preprocessing

#Calculate w for a given b and x
def calc_w(X, b):
    return 1/(1 + np.exp(- X * b))

#Calculate the slope of the cost function for a single data point, i
def calc_gradient(X, y, b):
    lam = 0.1
    w = calc_w(X, b)
    return X.T * (w-y)/X.shape[0] + lam * (b > 0).astype(float) - lam * (b < 0).astype(float)
    #return X.T * (w-y) #np.dot(X.transpose(), (w-y))

#Calculate likliehood function
def calc_l(X, y, b):
    lam = 0.1
    w = calc_w(X, b)
    #l = -(np.nan_to_num(np.dot(y.transpose(), np.log(w)))+np.nan_to_num(np.dot((1-y).transpose(), np.log(1-w))))
    l = -(np.dot(y.T, np.log(w + 1e-7)) + np.dot((1 - y).T, np.log(1-w + 1e-7)))/X.shape[0] +  lam * np.linalg.norm(b, 1)
    #print l
    return l

#Check the convergance of the log likliehood function within some criteria, e. Return true if so
def test_converge(l, e):
    if len(l)>5: 
        conv = (l[len(l)-1]-l[len(l)-2])#/l[len(l)-1]
        print conv
        if abs(conv) < e:
            return True
        else:
            return False
    else:
        return False 

#shuffle data after each epoch, return updated X, y
def shuffle(X, y):
    p = np.random.permutation(len(y))
    return X.shape[p], y[p]

#Perform gradient descent on b. Give the likliehood at each step.  Select a single data point each time, shuffle after running through entire set
def run_descent(X, y, b, e, step_size, l):
    batch_size = 1000
    hist_grad=0
    conv = False
    for i in xrange(len(y)/batch_size):
        if (i+1)*batch_size < len(y):
            X_temp = X[i*batch_size:(i+1)*batch_size]
            y_temp = y[i*batch_size:(i+1)*batch_size]
            #b_temp = b[i*batch_size:(i+1)*batch_size]
        else:
            X_temp = X[i*batch_size:]
            y_temp = y[i*batch_size:]
            #b_temp = b[i*batch_size:]

            #for j in xrange(10):
            #for i,k in zip(X_bat.col, X_bat.data):
            #j = i%len(y)
            #if j==0:
            #    X, y = shuffle(X, y)
        
        grad_b = calc_gradient(X_temp, y_temp, b)
        hist_grad += np.square(grad_b)
        b -= grad_b*step_size*(1/np.sqrt(hist_grad + e))
        #b[i*batch_size:(i+1)*batch_size] = b_temp
        l.append(calc_l(X, y, b))
        #j+=1 
        conv = test_converge(l, e)
        if conv:
            break
    return b, l, conv

#Test if the b found correctly predicts B or M, given X, y and calculated b. Return fraction of success
def predict_acc(X, y, b):
    w_pred = calc_w(X, b)
    n_correct = 0
    p = []

    xb = np.dot(X, b)
    #xb = xb/np.linalg.norm(xb, ord=1)
    w_pred = 1/(1 + np.exp(-xb))

    for i in range(len(w_pred)):
        if w_pred[i] > 0.5:
            p.append(1)
        else:
            p.append(0)

        if p[i]==y[i]:
            n_correct+=1

    return float(n_correct)/float(len(y))

#Read in svm files one at a time. Return the X and y arrays that result
inFiles = open('svm_files.txt', 'r')

def read_file(f):
    f = f.rstrip()
    print "Reading file: "+f
    X,y = sk.datasets.load_svmlight_file(f, n_features = 3231962)
    ones = np.ones((len(y),1))
    ones = scipy.sparse.csr_matrix(ones)
    X = scipy.sparse.hstack([X, ones])
    X = X.tocsr()
    return X, y
    
#Initialize b and l vectors
b = np.random.rand(3231963)
l_tot = []

#Run adagrad, looping over the files
for f in inFiles:
    X, y = read_file(f)
    l = []
    b, l, conv = run_descent(X, y, b, 0.001, 0.01, l)
    print conv
    if conv:
        break
    l_tot = l_tot+l
    
    X = None
    y = None
    l = None
    
plt.figure()
plt.plot(l_tot)

plt.ylabel('-log likliehood')
plt.xlabel('Iteration')
plt.savefig('result_better_sgd.png', format = 'png')

#b_tot = b_tot/np.linalg.norm(b_tot, ord=1)
#print "Step Size: "+str(step_size)
#print "Accuracey of prediction: "+str(predict_acc(X, y, b))

#b = None 

