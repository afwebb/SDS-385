# Week 5 Exercises

## Penalized Likliehood

A.) 

B.) The code for this excercise can be found [here](penalized.py)

A random y and sparse theta are generated. An S vector is calculated based on the vector. The resulting mean square error as a function of lambda is shown below, for several different levels of sparsity:

<img src="https://github.com/afwebb/SDS-385/blob/master/week5/result_pen.png" width="500">

The result suggests that for a sparse vector, a higher lambda greatly improves the estimate. However, as the sparsity decreases, a higher lambda helps less and less, and actually hurts the fit overall for more dense vectors.

## The Lasso

The code for this excercise can be found [here](lasso.py)

<img src="https://github.com/afwebb/SDS-385/blob/master/week5/result_lasso.png" width="500">
