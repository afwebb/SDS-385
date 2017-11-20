import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import datasets

#Calculate the weight of each point using the gaussian kernel
def calc_weight(S, y, b):
    w_vec = []
    for i in xrange(len(y)):
        w = 0
        for j in xrange(len(y)):
            dist = np.abs(S[i,0]-S[j,0])+np.abs(S[i,1]-S[j,1])
            w+=np.max(0, 3/(4*b)*(1-(dist**2)/(2*b**2)))
        w_vec.append(w)
    return w_vec

#Calculate the x vector from the y vector, and the weight vector
def calc_x(w, y):
    return w*y/np.sum(w)

#Read in the S and y matrices. Keep the day info too
dataSet = pd.read_csv('co2.csv')
S = dataSet[["lon","lat"]].values
y = dataSet["co2avgret"].values
days = dataSet["day"].values

#Calculate the w vector, and the x vector
w = calc_weight(S, y, 1)
x = calc_x(w, y)

#Plot the data before the smoothing
plt.figure(0)
plt.scatter(S[:,0], S[:,1], c=y[:], marker=".", cmap=plt.cm.coolwarm)
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.savefig('unsmoothed.png', format = 'png')

#Plot the data after the smoothing
plt.figure(1)
plt.scatter(S[:,0], S[:,1], c=x[:], marker=".", cmap=plt.cm.coolwarm)
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.savefig('smoothed.png', format = 'png')
