import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Calculate w for a given b and x
def calc_w(X, b):
    return 1/(1 + np.exp(-np.dot(X, b)))

#Calculate the (vector) slope of the cost function for a given X, y and b
def calc_slope(X, y, b):
    w = calc_w(X, b)
    return np.dot(X.transpose(), (w-y))

#Calculate likliehood function
def calc_l(X, y, b):
    w = calc_w(X, b)
    l = -(np.dot(y.transpose(), np.log(w))+np.dot((1-y).transpose(), np.log(1-w)))
    return l

#Take the matrix X, vectors y and b, an integer number of iterations and a float step size. Returns a fit value of b. Give the likliehood at each step
def run_descent(X, y, b, num_iter, step_size, l):
    for i in range(num_iter):
        b -= calc_slope(X, y, b)*step_size
        l.append(calc_l(X, y, b))
    return b, l

#Test if the b found correctly predicts B or M, given X, y and calculated b. Return fraction of success
def predict_acc(X, y, b):
    w_pred = calc_w(X, b)
    n_correct = 0
    p = []

    xb = np.dot(X, b)
    xb = xb/np.linalg.norm(xb, ord=1)
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
X = X/np.linalg.norm(X, ord=1)

b = np.random.rand(len(X[0]))
b = b/np.linalg.norm(b, ord=1)

#Run deepest descent for a few different numbers of iterations. Plot the l that results. Print the accuracy of the calculated b vector
for iter in [100, 1000, 10000, 100000]:
    l = []
    b, l = run_descent(X, y, b, iter, 3000, l)

    plt.figure()
    plt.plot(l)

    plt.ylabel('-log likliehood')
    plt.xlabel('Iteration')
    plt.savefig('deepest_descent_'+str(iter)+'.pdf', format = 'pdf')

    b = b/np.linalg.norm(b, ord=1)
    print "Number of iterations: "+str(iter)
    print "Accuracey of prediction: "+str(predict_acc(X, y, b))

