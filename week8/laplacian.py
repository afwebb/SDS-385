import scipy
from scipy import sparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import time
from sklearn import datasets
from sklearn import neighbors
from sklearn.neighbors.kde import KernelDensity
from scipy.sparse.linalg import LinearOperator

#Calculate the x vector from the y vector, and the weight vector
def calc_l(S, k):
    A = neighbors.kneighbors_graph(S, k, mode='connectivity', metric='euclidean', p=2, n_jobs=-1)
    W = A.sum(1)
    W = np.diagflat(W)
    W = sparse.csr_matrix(W)
    return W - A

#Plot the co2 concentration
def plot_result(S, name, num):
    plt.figure(num)
    #plt.scatter(lon, lat, c=y, s=0.1, edgecolor="face", vmin=360, vmax=400)
    plt.imshow(S, interpolation='nearest')#, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig(name+'.png', format = 'png')

#Read in the S and y matrices. Keep the day info too
dataSet = pd.read_csv('fmri_z.csv')
S = dataSet.values
n = len(S[0])
y = np.random.rand(n)

#Perform kernel density estimation for several weighting functions. Time the results
time0 = time.time()
L = calc_l(S, 1)
L2 = scipy.sparse.csgraph.laplacian(S)
print L2[50,:]
print S[50,:]
time1 = time.time()

lam = 0.1
ident = sparse.identity(len(S[0]))
L_inv = ident - L2
print L_inv.shape
L_inv = sparse.linalg.inv(ident)
S_smoothed = L_inv.dot(S)

#Plot the results for each algorithm
plot_result(S, "heatmap_unsmoothed", 0)
plot_result(S_smoothed, "heatmap_smoothed", 1)
