#Designed to read in each data file and output a single data matrix

import scipy 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import datasets
from tempfile import TemporaryFile

#Read in csv files one at a time as X. Have y be a series of ones for ttH sample, zero for backgrounds
for dsid in ['343365', '410155', '410218', '410219']:#,'363491']:
    print "Adding " +dsid
    X_temp = pd.read_csv(dsid+".csv")

    if dsid == '343365':
        y_temp = np.ones(X_temp.shape[0])
    else:
        y_temp = np.zeros(X_temp.shape[0])

    try:
        print X.shape
        X = pd.concat([X,X_temp])#scipy.sparse.vstack([X,X_temp])
        print X.shape
        y = scipy.concatenate([y,y_temp])
    except NameError:
        X = X_temp.copy()
        y = y_temp.copy()

X = scipy.sparse.csr_matrix(X.values)

#Add a column of ones. Shuffle X and y
ones = np.ones((len(y),1))
ones = scipy.sparse.csr_matrix(ones)
X = scipy.sparse.hstack([X, ones])
X,y = sk.utils.shuffle(X,y)

#Save the result
sk.datasets.dump_svmlight_file(X, y, 'input_data')
