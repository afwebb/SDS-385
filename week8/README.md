# Spatial Smoothing

The code for this section is [here](spatial_smoothing.py)

Here's how the data looks unsmoothed:

<img src="https://github.com/afwebb/SDS-385/blob/master/week8/unsmoothed.png" width="500">

I originally tried to do the smoothing with dense matrices, looping over the data and performing gaussian smoothing. This took about an hour and gave the following result:

<img src="https://github.com/afwebb/SDS-385/blob/master/week8/gaussian.png" width="500">

I then used scikitlearn's neighbor mapping, which produces a sparse distance matrix. Since exponentiating ruined my sparsity, I used the Epanechnikov kernel instead of the gaussian. The code looks like this:

```python

```

This went way faster than  I did this for two different values of bandwidth:

```python
Gaussian time: 2369.64 s
Linear time with 10 nearest neighbors: 2.24 s
Linear time with 500 nearest neighbors: 92.94 s
```

The result for a bandwidth of 1:

<img src="https://github.com/afwebb/SDS-385/blob/master/week8/linear_10.png" width="500">

And for a bandwidth of 100:

<img src="https://github.com/afwebb/SDS-385/blob/master/week8/linear_500.png" width="500">
