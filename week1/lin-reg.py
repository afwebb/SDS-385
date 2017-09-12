import numpy as np
import time

#Define the size of the matrix
n=10
p=10

#Generate x, Y, and w matrices, filled with random data
X = np.random.random((n, p))
Y = np.random.random(n)
W = np.random.random((p,n))

XWX = np.dot(X.transpose(), np.dot(W, X))

#Solve for beta using inverse matrix method
def inv_solve(X, Y, W):
    inv_XWX = np.linalg.inv(XWX)
    return np.dot(inv_XWX, np.dot(X.transpose(), Y))

#Solve for beta using qr matrix decomposition
def comp_solve(X, Y, W):
    q,r = np.linalg.qr(XWX)
    #Here inv(XWX) = inv(r)trans(q)
    return np.dot(np.dot(np.linalg.inv(r), q), np.dot(X.transpose(), Y))



b_inv = inv_solve(X, Y, W)
b_qr = comp_solve(X, Y, W)

    

