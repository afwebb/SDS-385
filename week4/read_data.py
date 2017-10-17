import scipy 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import datasets
from tempfile import TemporaryFile

#Read in svm files one at a time. Save the X and y arrays that result
inFiles = open('svm_files.txt', 'r')

for f in inFiles:
    f = f.rstrip()
    print "Reading file: "+f
    X_temp,y_temp = sk.datasets.load_svmlight_file(f, n_features = 3231962)
    try:
        X = scipy.sparse.vstack([X,X_temp])
        y = scipy.concatenate([y,y_temp])
    except NameError:
        X,y = sk.datasets.load_svmlight_file(f, n_features = 3231962)
    X_temp = None
    y_temp = None

print X.shape
ones = np.ones((len(y),1))
ones = scipy.sparse.csr_matrix(ones)
X = scipy.sparse.hstack([X, ones])
X,y = sk.utils.shuffle(X,y)

sk.datasets.dump_svmlight_file(X, y, 'data')
#np.savez_compressed('data_temp', y=y)
#scipy.sparse.save_npz('data_X', X)
