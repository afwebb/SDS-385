import numpy as np
import matplotlib.pyplot as plt

import sklearn as sk
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

def read_file(f, i, batch_size):
    X,y = sk.datasets.load_svmlight_file(f)# offset=i*batch_size, length=batch_size)
    X = X.tocsr()
    return X, y

X, y = read_file("data", 0, 10)

#X = sk.preprocessing.normalize(X, norm='l1', axis=1)
print X
