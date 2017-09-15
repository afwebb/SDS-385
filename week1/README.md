# Week 1 Excercises

## Linear Regression

A.) The minimum occurs when the gradient of the function is equal to zero.
So, d/dB (W/2)(y-X_t*B)^2 = 0

= X_t W y - X_t W X B = 0
  And the result follows
  
B.) Inverting matrices is often computationally intensive. One way to simplify the calculation is to decompose a complex matrix into
multiple, easier to handle matrices. One such method is QR decomposition. This splits a matrix, A, into a orthogonal matrix, Q, 
and an upper triangular matrix R such that A = QR.

Inv(A) = Inv(R)Inv(Q)= Inv(R)Trans(Q). Calculating Inv(R) is simple, as it is upper triangle. The code would look like this:

```python
def invert(X, W):
  A = trans(X)WX
  q, r = decomp_qr(A)
  return inv(q).trans(r)
```

C-D.) [The code for this part is here](week1/ex1.py)

The results for part D) look like this, for several levels for sparsity:

```python
z=0.5
Inverse solve took: 3.78789520264
QR Decomposition Solve took: 6.93567991257
Sparse Solver: 0.511936903

z=0.1
Inverse solve took: 3.80782318115
QR Decomposition Solve took: 5.97365999222
Sparse Solver: 0.372360944748

z=0.05
Inverse solve took: 4.13837814331
QR Decomposition Solve took: 5.5008699894
Sparse Solver: 0.325119018555
```

## Generalized Linear Models

A.) [Algebra for this problem is here](image.png)

B.) [The code for this part is here](week1/ex2.py)

The likliehood function for several numbers of iterations look like this:

<img src="https://github.com/afwebb/SDS-385/blob/master/week1/deepest_descent_1000.png" width="500">

```python
Accuracy of prediction: 0.817
```

<img src="https://github.com/afwebb/SDS-385/blob/master/week1/deepest_descent_100000.png" width="500">

```python
Accuracy of prediction: 0.878
```
I ended up getting the step size through trial and error. I'd like to figure a better way, and use variable step size, but I haven't had time yet.

I also am not convinced my W matrix represents a real probability. In order to get reasonable answers, I normalized my X.b matrix, but this suspect. If I got that right, I might be able to do better than ~88% prediction rate.

