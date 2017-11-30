import scipy
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
    W = scipy.sparse.csr_matrix(W)
    return W - A

#Plot the co2 concentration
def plot_result(S, name, num):
    plt.figure(num)
    #plt.scatter(lon, lat, c=y, s=0.1, edgecolor="face", vmin=360, vmax=400)
    plt.imshow(S)#, cmap='hot', interpolation='nearest')
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
print L
L2 = scipy.sparse.csgraph.laplacian(S)
print L2
time1 = time.time()

print "L time: "+str(time1-time0)

#Plot the results for each algorithm
plot_result(S, "heatmap", 0)
