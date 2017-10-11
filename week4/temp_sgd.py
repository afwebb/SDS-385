import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import datasets
from sklearn import preprocessing

#Calculate w for a given b and x
def calc_w(X, b):
    return 1/(1 + np.exp(- X.T * b))

#Calculate the slope of the cost function for a single data point, i
def calc_gradient(X, y, b):
    w = calc_w(X, b)
    print w.shape
    print y.shape
    return X.T * (w-y)/X.shape[0] #np.dot(X.transpose(), (w-y))

#Calculate likliehood function
def calc_l(X, y, b):
    w = calc_w(X, b)
    l = -(np.nan_to_num(np.dot(y.transpose(), np.log(w)))+np.nan_to_num(np.dot((1-y).transpose(), np.log(1-w))))
    #print l
    return l

#Check the convergance of the log likliehood function within some criteria, e. Return true if so
def test_converge(l, e):
    if len(l)>5: 
        conv = (l[len(l)-1]-l[len(l)-2])/l[len(l)-1]
        #print conv
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

    X_batches, y_batches, b_batches = split_data(X, y, b)
    b_result = []

    #for i in range(e):
    j = 0
    for i in xrange(len(X_batches)):
        X_bat = X_batches[i]
        y_bat = y_batches[i]
        b_bat = b_batches[i]
        hist_grad=0
        while not test_converge(l, e):
            #for i,k in zip(X_bat.col, X_bat.data):
            #j = i%len(y)
            #if j==0:
            #    X, y = shuffle(X, y)
            
            grad_b = calc_gradient(X_bat, y_bat, b_bat)
            hist_grad += np.square(grad_b)
            b_bat -= grad_b*step_size*(1/np.sqrt(hist_grad + e))
            l.append(calc_l(X_bat, y, b_bat))
            #j+=1 
        b_result = np.concatenate(b_result, b_bat)
    return b_result, l

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

#Split data into batches
def split_data(X, y, b, batch_size=10000):
    X_batches = []
    y_batches = []
    b_batches = []
    
    for i in xrange(len(y)/batch_size):
        if (i+1)*batch_size < len(y):
            X_temp = X[i*batch_size:(i+1)*batch_size]
            y_temp = y[i*batch_size:(i+1)*batch_size]
            b_temp = b[i*batch_size:(i+1)*batch_size]
        else:
            X_temp = X[i*batch_size:]
            y_temp = y[i*batch_size:]
            b_temp = b[i*batch_size:]

        X_batches.append(X_temp)
        y_batches.append(y_temp)
        b_batches.append(b_temp)

    return X_batches, y_batches, b_batches


X = scipy.sparse.load_npz('data_X.npz')
npzFile = np.load('data_y.npz')
y = npzFile['y']
X = X.tocsr()

b_orig = np.random.rand(X.shape[1])

#Run stocastic descent for different step sizes. Plot likliehood, and accuracey of prediction
l = []
b = b_orig.copy()

b, l = run_descent(X, y, b, 0.0001, 0.01, l)

plt.figure()
plt.plot(l)

plt.ylabel('-log likliehood')
plt.xlabel('Iteration')
plt.savefig('result_qnewton.png', format = 'png')

b = b/np.linalg.norm(b, ord=1)
print "Step Size: "+str(step_size)
print "Accuracey of prediction: "+str(predict_acc(X, y, b))

b = None 

