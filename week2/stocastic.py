import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

#Calculate w for a given b and x
def calc_w(X, b):
    return 1/(1 + np.exp(-np.dot(X, b)))

#Calculate the slope of the cost function for a single data point, i
def calc_gradient(X, y, b, i):
    w = calc_w(X[i], b)
    return np.dot(X[i], (w-y[i]))

#Calculate likliehood function
def calc_l(X, y, b):
    w = calc_w(X, b)
    l = -(np.dot(y.transpose(), np.log(w))+np.dot((1-y).transpose(), np.log(1-w)))
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
    return X[p], y[p]

#Perform gradient descent on b. Give the likliehood at each step.  Select a single data point each time, shuffle after running through entire set
def run_descent(X, y, b, e, step_size, l):
    #for i in range(e):
    i = 0
    while not test_converge(l, e):
        j = i%len(y)
        if j==0:
            X, y = shuffle(X, y)
        b -= calc_gradient(X, y, b, j)*step_size
        l.append(calc_l(X, y, b))
        i+=1
        
        if i>1000000:
            break

    return b, l

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
        
#Read in the data set 
dataSet = pd.read_csv('wdbc.csv',header=None, usecols=range(1,12))

#Define X matrix and y vector from the data
#Select the first column for y, convert to binary array
y = dataSet[1].map({'M':1, 'B':0})
y = y.values
#Use the rest for X. Scale X by the size of the vector
X = dataSet.drop(1, 1).values
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
#X = X/np.linalg.norm(X, ord=1)

b_orig = np.random.rand(len(X[0]))
#b_orig = b_orig/np.linalg.norm(b_orig, ord=1)

#Run deepest descent for a few different numbers of iterations. Plot the l that results. Print the accuracy of the calculated b vector
for step_size in [0.01, 0.1, 1]:
    l = []

    b, l = run_descent(X, y, b_orig, 0.00001, step_size, l)

    plt.figure()
    plt.plot(l)

    plt.ylabel('-log likliehood')
    plt.xlabel('Iteration')
    plt.savefig('result_'+str(step_size)+'.png', format = 'png')

    b = b/np.linalg.norm(b, ord=1)
    print "Step Size: "+str(step_size)
    print "Accuracey of prediction: "+str(predict_acc(X, y, b))
