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
print score
print bdt.feature_importances_

# Plot the two-class decision scores
output = bdt.decision_function(X_test)
score_sig = output[y_test == 1]
score_bkg = output[y_test == 0]
print score_sig.shape
print score_bkg.shape

#print output.shape
plt.figure(0)
#plt.hist(pred_sig, label='signal')
#plt.hist(pred_bkg, label='background')
plt.hist([score_sig, score_bkg])
plt.legend(loc='upper right')
plt.savefig('probability.png', format='png')

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
pred_y = bdt.predict_proba(X_test)
#test_pred = zip(y_test, y_pred)
pred_sig = pred_y[y_test==1]
pred_bkg = pred_y[y_test==0]

plt.figure(0)
#plt.hist(pred_sig, label='signal')
#plt.hist(pred_bkg, label='background')
plt.hist([pred_y, pred_bkg])
plt.legend(loc='upper right')
plt.savefig('probability.png', format='png')

'''

def read_file(f, i, batch_size):
    X,y = sk.datasets.load_svmlight_file(f)# offset=i*batch_size, length=batch_size)
    X = X.tocsr()
    return X, y

X, y = read_file("input_data", 0, 10)

#X = sk.preprocessing.normalize(X, norm='l1', axis=1)

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)

bdt.fit(X, y)

plot_colors = "br"
plot_step = 0.02
class_names = "AB"

plt.figure(figsize=(10, 5))

# Plot the decision boundaries
plt.subplot(121)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis("tight")

# Plot the training points
for i, n, c in zip(range(2), class_names, plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1],
                c=c, cmap=plt.cm.Paired,
                s=20, edgecolor='k',
                label="Class %s" % n)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Boundary')

# Plot the two-class decision scores
twoclass_output = bdt.decision_function(X)
plot_range = (twoclass_output.min(), twoclass_output.max())
plt.subplot(122)
for i, n, c in zip(range(2), class_names, plot_colors):
    plt.hist(twoclass_output[y == i],
             bins=10,
             range=plot_range,
             facecolor=c,
             label='Class %s' % n,
             alpha=.5,
             edgecolor='k')
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, y1, y2 * 1.2))
plt.legend(loc='upper right')
plt.ylabel('Samples')
plt.xlabel('Score')
plt.title('Decision Scores')

plt.tight_layout()
plt.subplots_adjust(wspace=0.35)
plt.show()

