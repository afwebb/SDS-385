import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import time
from sklearn import datasets
from sklearn.neighbors.kde import KernelDensity

#Calculate the x vector from the y vector, and the weight vector
def calc_x(S, kde, y):
    w = kde.score_samples(S)
    return w*y/np.sum(w)

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

kde_gaus = KernelDensity(kernel='gaussian', bandwidth=5).fit(S)
x_gaus = calc_x(S, kde_gaus, y)

time1 = time.time()

kde_tophat = KernelDensity(kernel='tophat', bandwidth=5).fit(S)
x_tophat = calc_x(S, kde_tophat, y)

time2 = time.time()

kde_linear = KernelDensity(kernel='linear', bandwidth=5).fit(S)
x_linear = calc_x(S, kde_linear, y)

time3 = time.time()

print "Gaussian time: "+str(time1-time0)
print "Tophat time: "+str(time2-time1)
print "Linear time: "+str(time3-time2)

#Plot the results for each algorithm
plot_result(y, "unsmoothed", 1)
plot_result(x_gaus, "gaussian", 2)
plot_result(x_tophat, "tophat", 3)
plot_result(x_linear, "linear", 4)
