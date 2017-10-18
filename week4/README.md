# Week 4 Exercises

## Better SGD

The code for this excercise can be found [here](better_sgd.py)

Reading in the data took much longer than anything else, so (completely stealing the idea from Bowen) I wrote a [script](read_data.py) to extract the information from the 120 files and save the result. 

I use scikitlearn's load_svm_light_file and dump_svm_file to extract the data and save it to a file that can be acccessed quickly. 

The dataset is so big I ran into memory problems loading it all. So rather than loading in all the data at once, I read in a small subset of the data at a time, and perform adagrad on that subset. I keep track of the likliehood function, and repeat the process until convergence is achieved. The code looks like this:

```python
while not test_converge(l, e):
        X,y= read_file('data', i, batch_size) #Read in the data a little at a time
        if i%20==0:
            step_size=0.5*step_size #decrease step size over time
            print i
        grad_b = calc_gradient(X, y, b) 
        hist_grad += np.square(grad_b) #update hist grad
        b -= grad_b*step_size*(1/np.sqrt(hist_grad + e)) #update b
        l.append(calc_l(X, y, b)) #track l
```

I read in 5000 entries at a time, and decreased the step size by hand over time (my attempts at variable step size haven't panned out). This allowed me to start with a larger step size while still demanding a strict convergence criterion.

A penalty term of 0.1 is applied to the gradient, as well as the likliehood function.

This batch gradient descent approach converged much quicker than using only 1 entry at a time, and certainly quicker than full gradient descent.

<img src="https://github.com/afwebb/SDS-385/blob/master/week4/result_sgd.png" width="500">

Convergence is achieved after 38 seconds. 

Accuracey of the resulting beta is tested on a subset of the data (10,000,000 entries). The accuracey of the prediction is found to be 98%
