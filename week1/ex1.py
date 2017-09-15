import numpy as np
import scipy
from scipy.sparse import csr_matrix 
from scipy.sparse.linalg import spsolve
import pandas as pd
import time

#Define the size of the matrix
n=1000
p=1000

#Generate x, Y, and w matrices, filled with random data
X = np.random.random((n, p))
Y = np.random.random(n)
W = np.random.random((n,n))

#Solve for vectr b for weighted y=Xb using inverse matrix method given X nXp matrix, W nXn matrix, and Y vector
def inv_solve(X, Y, W):
    XWX = np.matmul(X.transpose(), np.matmul(W, X))
    inv_XWX = np.linalg.inv(XWX)
    return np.matmul(inv_XWX, np.matmul(X.transpose(), Y))
    
#Solve for beta using qr matrix decomposition
def comp_solve(X, Y, W):
    XWX = np.matmul(X.transpose(), np.matmul(W, X))
    q,r = np.linalg.qr(XWX)
    #Here inv(XWX) = inv(r)trans(q)
    return np.matmul(np.matmul(np.linalg.inv(r), q.transpose()), np.matmul(X.transpose(), Y))

#Solve using sparse algorithms
def sparse_solve(X, Y, W):
    X = csr_matrix(X)
    return spsolve(X, Y)

#calculate the time to carry out function f
def time_solve(f, X, Y, W):
    start_inv = time.time()
    res = f(X, Y, W)
    return res, time.time()-start_inv

#Run the solve
b_inv, inv_time = time_solve(inv_solve, X, Y, W)
b_qr, qr_time = time_solve(comp_solve, X, Y, W)

#check if solutions match, output time if so
if b_inv.all()==b_qr.all():
    print "Solutions match!"
    print "Inverse solve took: "+str(inv_time)
    print "QR Decomposition Solve took: "+str(qr_time)
else:
    print "Solutions don't match..."

#Now try with sparse data
#Repeat for several levels of sparsity 
for z in [0.5, 0.1, 0.05]:
    #Define sparse matrix
    X = np.random.choice([0, 1], size=(n,p), p=[1-z, z])
    #X = csr_matrix(X)
    
    #Replace weight's with 1 for simplicity
    W = np.identity(n)

    #Solve using each of the three methods
    b_inv, inv_time = time_solve(inv_solve, X, Y, W)
    b_qr, qr_time = time_solve(comp_solve, X, Y, W)
    b_sq, sq_time = time_solve(sparse_solve, X, Y, W)

    #Display Results
    print "z="+str(z)
    print "Inverse solve took: "+str(inv_time)
    print "QR Decomposition Solve took: "+str(qr_time)
    print "Sparse Solver: "+str(sq_time)
    print " " 
    
