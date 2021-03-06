import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import datasets
from sklearn import preprocessing
from time import time

#Calculate w for a given b and x
def calc_w(X, b):
    return 1/(1 + np.exp(- X * b))

#Calculate the slope of the cost function for a single data point, i
def calc_gradient(X, y, b):
    lam = 0.1
    w = calc_w(X, b)
    return X.T * (w-y)/X.shape[0] + lam * (b > 0).astype(float) - lam * (b < 0).astype(float) #Include penalty term on the gradient

#Calculate likliehood function
def calc_l(X, y, b):
    lam = 0.1
    w = calc_w(X, b)
    l = -(np.dot(y.T, np.log(w + 1e-7)) + np.dot((1 - y).T, np.log(1-w + 1e-7)))/X.shape[0] +  lam * np.linalg.norm(b, 1) #include penalty term on the loss function
    return l

def calc_batch_size(b):

    calc_times = []
    l_change_vec = []
    i = 10
    batch_values = []
    batch_max = 0

    while i < 10000:
        start = time()
        X,y= read_file('data', 0, i)
        l1 = calc_l(X,y,b)
        b -= 0.1*calc_gradient(X, y, b)
        l2 = calc_l(X,y,b)
        end = time()

        grad_time = end-start
        l_change = l1-l2
        calc_times.append(grad_time)
        l_change_vec.append(l_change)
        batch_values.append(i)
        

        if l_change/grad_time > batch_max:
            batch_max = abs(l_change/grad_time)

        i=i*1.5

    plt.figure(2)
    plt.semilogx(batch_values, calc_times)
    plt.ylabel('Calculation Time')
    plt.xlabel('log Batch Size')
    plt.savefig('time_vs_batch.png', format = 'png')

    plt.figure(3)
    plt.semilogx(batch_values, l_change_vec)
    plt.ylabel('Change in l')
    plt.xlabel('Batch Size')
    plt.savefig('changeL_vs_batch.png', format = 'png')

    r = np.array(calc_times)/np.array(l_change_vec)
    plt.figure(4)
    plt.semilogx(batch_values, r)
    plt.ylabel('Change in l / Calculation Time')
    plt.xlabel('Batch Size')
    plt.savefig('ratio_vs_batch.png', format = 'png')

    return batch_max

#Check the convergance of the log likliehood function within some criteria, e. Return true converged
def test_converge(l, e):
    if len(l)>5: 
        conv = (l[len(l)-1]-l[len(l)-2])/l[len(l)-1]
        conv2 = (l[len(l)-2]-l[len(l)-3])/l[len(l)-2] 
        if abs(conv) < e and abs(conv2) < e: #only return true if two successive entries meet criteria. Reduces chance of ending adagrad before converging
            return 1
        else:
            return 0
    else:
        return 0 

#Perform stocastic gradient descent on b. Give the likliehood at each step.
def run_descent(X, y, b, e, step_size, batch_size, l):

    hist_grad=0 #track gradient
    i = 0 #track iteration count

    while not test_converge(l, e):
        X,y= read_file('data', i, batch_size) #Read in the data a little at a time
        if i%20==0:
            step_size=0.5*step_size #decrease step size over time
            print i
        grad_b = calc_gradient(X, y, b) 
        hist_grad += np.square(grad_b) #update hist grad
        b -= grad_b*step_size*(1/np.sqrt(hist_grad + e)) #update b
        l.append(calc_l(X, y, b)) #track l
        i+=1 

    return b, l 

#Test if the b found correctly predicts B or M, given X, y and calculated b. Return fraction of success
def predict_acc(b):
    X,y = read_file('data', 0, 10000000) #Read in a section of the file
    X = sk.preprocessing.normalize(X)
    w_pred = calc_w(X, b)
    n_correct = 0
    p = []

    for i in range(len(y)):
        if w_pred[i] > 0.5:
            p.append(1)
        else:
            p.append(0)

        if p[i]==y[i]:
            n_correct+=1

    return float(n_correct)/float(len(y))

#Read in svm files one at a time. Return the X and y arrays that result
inFiles = open('svm_files.txt', 'r')

def read_file(f, i, batch_size):
    X,y = sk.datasets.load_svmlight_file(f, n_features = 3231963, offset=i*batch_size, length=batch_size)
    X = X.tocsr()
    return X, y
    
#Initialize b and l vectors
b = np.random.rand(3231963)
X = []
y = []
l=[]

#Determine optimal batch size
batch_size = 500#calc_batch_size(b)
print batch_size

#Run adagrad, looping over the files
start = time()
b, l = run_descent(X, y, b, 0.0001, 0.1, batch_size, l)
end = time()

print "Time to complete: "+str(end-start)

plt.figure()
plt.plot(l)

plt.ylabel('-log likliehood')
plt.xlabel('Iteration')
plt.savefig('result_better_sgd.png', format = 'png')

#b_tot = b_tot/np.linalg.norm(b_tot, ord=1)
print "Accuracey of prediction: "+str(predict_acc(b))

