import numpy as np
import matplotlib.pyplot as plt

import sklearn as sk
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

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
X, y = read_file("test_data", 0, 10)

#Normalize X
#X = sk.preprocessing.normalize(X, norm='l1', axis=1)

#Define the BDT
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=50)

#Fit the BDT to data
X_train, y_train, X_test, y_test = cross_val(X, y, 10)

bdt.fit(X_train, y_train)
score = bdt.score(X_test, y_test)
print 'BDT score: '+str(score)
print "importance"
print bdt.feature_importances_

output = bdt.decision_function(X_test)
score_sig = output[y_test == 1]
score_bkg = output[y_test == 0]

significance = 0
for val in np.linspace(-0.8, 0.2, num=200):
    s = len(score_sig[score_sig > val] )
    b = len(score_bkg[score_bkg > val] )
    temp_sig = s/np.sqrt(b)
    if temp_sig > significance:
        significance = temp_sig

print "Significance" +str( significance ) 


pred_y = bdt.predict_proba(X_test)
#test_pred = zip(y_test, y_pred)
pred_sig = pred_y[y_test==1]
pred_bkg = pred_y[y_test==0]

plt.figure(0)
#plt.hist(pred_sig, label='signal')
#plt.hist(pred_bkg, label='background')
plt.hist([pred_y, pred_bkg])
plt.legend(loc='upper right')
plt.savefig('plots/bdt_probability.png', format='png')
