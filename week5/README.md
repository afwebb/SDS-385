# Week 5 Exercises

Summary of results for exercises 5

## Penalized Likliehood

The code for this excercise can be found [here](penalized.py)

A.) 

The following plots show equation 1 and S plotting for various values of lambda and y: 

<img src="https://github.com/afwebb/SDS-385/blob/master/week5/theta_s.png" width="500">

<img src="https://github.com/afwebb/SDS-385/blob/master/week5/theta_y.png" width="500">

The second plot in particular demonstrates how this function serves as a soft threshold, zeroing out more and more coefficients as lambda increases.

B.) 

A random y and sparse theta are generated. An S vector is calculated based on the vector. The resulting mean square error as a function of lambda is shown below, for several different levels of sparsity:

<img src="https://github.com/afwebb/SDS-385/blob/master/week5/result_theta.png" width="500">
<img src="https://github.com/afwebb/SDS-385/blob/master/week5/result_pen.png" width="500">

The result suggests that for a sparse vector, a higher lambda greatly improves the estimate. However, as the sparsity decreases, a higher lambda helps less and less, and actually hurts the fit overall for more dense vectors.

## The Lasso

The code for this excercise can be found [here](lasso.py)

A.)

<img src="https://github.com/afwebb/SDS-385/blob/master/week5/result_lasso_coef.png" width="500">
<img src="https://github.com/afwebb/SDS-385/blob/master/week5/coef_lambda.png" width="500">
As expected, as lambda increases, more and more of the coefficients are zeroed out. The value of the coefficiencts decrease as well.

B-C.)

<img src="https://github.com/afwebb/SDS-385/blob/master/week5/result_lasso_mse.png" width="500">

The lasso fit performs as expected over testing and training data. Splitting the data set into 5 or 10 parts didn't have a major impact over the ideal value of lambda, but did change the mse value more significantly on other parts of the curve.

The Mallow's Cp term yields results very similar to cross-validation, suggesting it can be used as an adequate alternative to cross validation.



