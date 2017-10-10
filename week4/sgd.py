import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import datasets

#Calculate w for a given b and x
def calc_w(X, b):
    return 1/(1 + np.exp(-np.dot(X, b)))

#Return a step size based on backtracking line search for a given loss function and search direction method
def find_step_size(X, y, b_in, f_loss, f_search):

    b = b_in.copy()
    rho = 0.8
    c = 0.01
    s_vec = []

    for i in xrange(len(b)):
        s = 1
        w = calc_w(X, b)
        l_new = f_loss(X[:,i], y, b[i]+s)
        l_old =  f_loss(X[:,i], y, b[i])
        l_comp =  f_loss(X[:,i], y, b[i]) + c*s*f_search(X[:,i], y, b[i])
        while l_new > l_comp: 
            s = rho*s
            b[i] -= s*f_search(X[:,i], y, b[i])
            l_new = f_loss(X[:,i], y, b[i]+s)
            l_comp = l_old + c*s*f_search(X[:,i], y, b[i])
        s_vec.append(s)
    return s_vec

#Calculate the slope of the cost function for a single data point, i
def calc_gradient(X, y, b):
    w = calc_w(X, b)
    return np.dot(X.transpose(), (w-y))

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
    return X[p], y[p]

#Perform gradient descent on b. Give the likliehood at each step.  Select a single data point each time, shuffle after running through entire set
def run_descent(X, y, b, e, step_size, l):
    #for i in range(e):
    i = 0
    while not test_converge(l, e):
        j = i%len(y)
        if j==0:
            X, y = shuffle(X, y)
        b -= calc_gradient(X[j], y[j], b)*step_size
        l.append(calc_l(X, y, b))
        i+=1
        
        if i>10000:
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
        
npzfile = np.load('data_temp.npz')
print npzfile.files
X = npzfile['X']
y = npzfile['y']
'''
y = y.values

#Use the rest for X. Scale X by the size of the vector
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
X = np.insert(X, 10, 1, axis=1)

b_orig = np.random.rand(len(X[0]))

#Run stocastic descent for different step sizes. Plot likliehood, and accuracey of prediction
l = []
b = b_orig.copy()

step_size = find_step_size(X, y, b, calc_l, calc_gradient)
b, l = run_descent(X, y, b, 0.0001, step_size, l)

plt.figure()
plt.plot(l)

plt.ylabel('-log likliehood')
plt.xlabel('Iteration')
plt.savefig('result_qnewton.png', format = 'png')

b = b/np.linalg.norm(b, ord=1)
print "Step Size: "+str(step_size)
print "Accuracey of prediction: "+str(predict_acc(X, y, b))

b = None 

'''
