import numpy as np
import matplotlib.pyplot as plt

import sklearn as sk
from sklearn.datasets import make_gaussian_quantiles
from sklearn.neural_network import MLPClassifier

#Read in the data file, f. Return feature matrix X, and data vector, y
def read_file(f, i, batch_size):
    X,y = sk.datasets.load_svmlight_file(f)#, n_features=None, length=-1)
    X = X.tocsr()
    return X, y

#Peform n-fold cross-validation. Return X_train, y_train, X_test, y_test
def cross_val(X, y, n):
    num = y.shape[0]/n
    
    X_test = X[:num]
    y_test = y[:num]
    
    X_train = X[num:]
    y_train = y[num:]

    return X_train, y_train, X_test, y_test

#Read in the data
X, y = read_file("input_data", 0, 10)

#Normalize X
#X = sk.preprocessing.normalize(X, norm='l1', axis=1)

#Define the MLP
mlp =  MLPClassifier()

#Fit the MLP to data
X_train, y_train, X_test, y_test = cross_val(X, y, 10)

mlp.fit(X_train, y_train)
score = mlp.score(X_test, y_test)
print score
'''
#print output.shape
#plt.figure(0)
#plt.hist(pred_sig, label='signal')
#plt.hist(pred_bkg, label='background')
#plt.hist([score_sig, score_bkg])
#plt.legend(loc='upper right')
#plt.savefig('probability.png', format='png')

significance = 0
for val in np.linspace(-0.8, 0.2, num=200):
    s = len(score_sig[score_sig > val] )
    b = len(score_bkg[score_bkg > val] )
    temp_sig = s/np.sqrt(b)
    if temp_sig > significance:
        significance = temp_sig
        print val

print significance
'''
