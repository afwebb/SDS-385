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

#Calculate the weight of each point using the gaussian kernel
def calc_gaussian(lon, lat, y, b):
    y_smoothed = []
    for i in xrange(len(y)):
        if i%1000==0:
            print i
        d = (lon-lon[i])**2 + (lat-lat[i])**2
        w = 1/b * np.exp(-d**2)/(2*b**2) 
        y_smoothed.append( np.dot(w, y)/np.sum(w) )
    return y_smoothed

#Calculate the x vector from the y vector, and the weight vector
def calc_linear(S, y, b, k):
    dist = neighbors.kneighbors_graph(S, k, mode='distance', metric='euclidean', p=2, n_jobs=-1)
    connect = neighbors.kneighbors_graph(S, k, mode='connectivity', metric='euclidean', p=2, n_jobs=-1)
    w = (b**2 * connect - dist.power(2))
    w = w.maximum(scipy.sparse.csr_matrix( (len(y), len(y) ) ) )
    w = sk.preprocessing.normalize(w, norm='l1', axis=1)
    y_smoothed = w.dot(y)
    return y_smoothed

#Plot the co2 concentration
def plot_result(y, name, num):
    plt.figure(num)
    plt.scatter(lon, lat, c=y, s=0.1, edgecolor="face", vmin=360, vmax=400)
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    plt.colorbar()
    plt.savefig(name+'.png', format = 'png')

#Read in the S and y matrices. Keep the day info too
dataSet = pd.read_csv('co2.csv')
S = dataSet[["lon","lat"]].values
lon = dataSet["lon"].values
lat = dataSet["lat"].values
y = dataSet["co2avgret"].values
days = dataSet["day"].values

#Perform kernel density estimation for several weighting functions. Time the results
time0 = time.time()
y_gaus = calc_gaussian(lon, lat, y, 5)

time1 = time.time()
y_linear_1 = calc_linear(S, y, 1, 50)
y_linear_5 = calc_linear(S, y, 500, 1000)
y_linear_20 = calc_linear(S, y, 50, 1000)

time2 = time.time()
#y_linear_500 = calc_linear(S, y, 5, 500)

time3 = time.time()

print "Gaussian time: "+str(time1-time0)
print "Linear time 10: "+str(time2-time1)
print "Linear time 500: "+str(time3-time2)

#Plot the results for each algorithm
plot_result(y, "unsmoothed", 1)
plot_result(y_gaus, "gaussian", 2)
plot_result(y_linear_1, "sparse_1", 3)
plot_result(y_linear_5, "sparse_test", 4)
plot_result(y_linear_20, "sparse_50", 5)


