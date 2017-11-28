# Spatial Smoothing

The code for this section is [here](spatial_smoothing.py)

Here's how the data looks unsmoothed:

<img src="https://github.com/afwebb/SDS-385/blob/master/week8/unsmoothed.png" width="500">

I originally tried to do the smoothing with dense matrices, looping over the data and performing gaussian smoothing. This took about an hour and gave the following result:

<img src="https://github.com/afwebb/SDS-385/blob/master/week8/gaussian.png" width="500">

I then used scikitlearn's neighbor mapping, which produces a sparse distance matrix. Since exponentiating ruined my sparsity, I used the Epanechnikov kernel instead of the gaussian. The code looks like this:

```python
def calc_linear(S, y, b, k):
    dist = neighbors.kneighbors_graph(S, k, mode='distance', metric='euclidean', p=2, n_jobs=-1)
    connect = neighbors.kneighbors_graph(S, k, mode='connectivity', metric='euclidean', p=2, n_jobs=-1)
    w = (b**2 * connect - dist.power(2))
    w = w.maximum(scipy.sparse.csr_matrix( (len(y), len(y) ) ) )
    w = sk.preprocessing.normalize(w, norm='l1', axis=1)
    y_smoothed = w.dot(y)
    return y_smoothed
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
