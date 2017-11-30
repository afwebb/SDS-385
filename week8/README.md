# Spatial Smoothing

The code for this section is [here](spatial_smoothing.py)

Here's how the data looks unsmoothed:

<img src="https://github.com/afwebb/SDS-385/blob/master/week8/unsmoothed.png" width="500">

I originally tried to do the smoothing with dense matrices, looping over the data and performing gaussian smoothing. 

```python 
def calc_gaussian(lon, lat, y, b):
    y_smoothed = []
    for i in xrange(len(y)):
        if i%1000==0:
            print i
        d = (lon-lon[i])**2 + (lat-lat[i])**2
        w = 1/b * np.exp(-d**2)/(2*b**2)
        y_smoothed.append( np.dot(w, y)/np.sum(w) )
    return y_smoothed
```

This took about an hour and gave the following result:

<img src="https://github.com/afwebb/SDS-385/blob/master/week8/linear_10.png" width="500">

I then used scikitlearn's neighbor mapping, which produces a sparse distance matrix for k nearest nighbors. There's also a version based on radius, but I couldn't get that working. Since exponentiating ruined my sparsity, I used the Epanechnikov kernel instead of the gaussian. The code looks like this:

```python
def calc_linear(S, y, b, k):
    dist = neighbors.kneighbors_graph(S, k, mode='distance', metric='euclidean', p=2, n_jobs=-1)
    connect = neighbors.kneighbors_graph(S, k, mode='connectivity', metric='euclidean', p=2, n_jobs=-1)
    w = b**2 * connect - dist.power(2)
    w = w.maximum(scipy.sparse.csr_matrix( (len(y), len(y) ) ) )
    w = sk.preprocessing.normalize(w, norm='l1', axis=1)
    y_smoothed = w.dot(y)
    return y_smoothed
```

This went way faster than the gaussian method: 

```python
Gaussian time: 3369.64 s
Linear time with 10 nearest neighbors: 2.24 s
Linear time with 500 nearest neighbors: 92.94 s
```

I scaled the number of nearest neighbors to the bandwidth to improve run times: 50 for b=1, 100 for b=5, and 1000 for b=50. Here are the result for a bandwidth of 1:

<img src="https://github.com/afwebb/SDS-385/blob/master/week8/sparse_1.png" width="500">

For a bandwidth of 5:

<img src="https://github.com/afwebb/SDS-385/blob/master/week8/sparse_5.png" width="500">

And for a brandwidth of 50:

<img src="https://github.com/afwebb/SDS-385/blob/master/week8/linear_500.png" width="500">
