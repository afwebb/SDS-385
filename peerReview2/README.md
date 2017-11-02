# Peer Review for Ashutosh

WIP

I couldn't find your solution for exercises 4 on your github.

## Exercises 5

### Written Solution

Your written solutions for the first part look good. You go into enough detail that I'm able to follow along with your logic, and I appreciate that you motivate the steps you're taking. The way you break up the differentiation into parts is well organized, but the answer at the very end seems a little sudden. An extra line or two about how you're combining the different regions would be nice.

### helper_new.py

I like how you defined all your functions in a seperate file like this. Its a clever way to do it, since a lot of these functions are reused between exercises. 

I particularly like you're create_dataset_logistic_bionomial function. This seems like a really useful thing to have for these kinds of problems, and its nice you've got it set up to produce arbitrary sized datasets. I think it might be useful to have another function to generate just b, since X and y are often given. Something like this:

```python
def gen_b(X):
   B = np.random.uniform(size=(X.shape[0],))
   return B
```

The functions are all pretty short, so its not hard to figure out what you're trying to do with each one. That said, a few more comments about what each function does would be good. In particular, stating where each function applies would probably be helpful. For example, you define this function

```python
def gradient(x,y,w) :
    gradient = (x.T.dot(w-y))/len(x)   ### why does it perform better?
    return gradient
```

Its easy to tell you're taking the gradient here, but the gradient of what? Since gradient is such a general thing I think it'd be useful to state "this function calculates the gradient for the loss function X". 

I'd also consider giving the function a more specific name so you reduce the chance of confusing it with this other function you've defined that's very similar.

```python
def gradient_regu(x,y,w,G,lamb) 
```

### IPYNB Notebook

#### Part A

Your plots of the hard and soft functions do a good job of demonstrating the general behavior, but I think they could be improved in a couple ways. For one, I'm not sure there's a good reason to use a scatter plot. A line plot would show the same thing without the gaps between points. Extending the x-axis to include negative makes more sense to me as well, to get across how the function is symmetric.

I also think you could plot functions for several different values of lambda on the same plot, so the general behavior can be seen all at once, rather than having to scroll through a half-dozen plots. I think taking plt.show() out of the for loop would do this. It would also be good if you included a label for each lambda value, as well as labels for your x and y axes. 

Here's how I would do it:

```python
for i in lam:
    Sy = th_hat_soft(y,i)
    Hy = th_hat_hard(y,i)
    
    plt.figure(1) #This seperates the two plots
    plt.plot(y, Sy, label="Lambda: "+str(lam))

    plt.figure(2) #Switch to hard plot
    plt.plot(y, Hy, label="Lambda: "+str(lam))
    
plt.figure(1)
plt.title('Soft Thresholding')
plt.legend(loc='upper left')
plt.show()

plt.figure(2)
plt.title('Hard Thresholding')
plt.legend(loc='upper left')
plt.show()

```

#### Part B

Most of my suggestions on part A I would say apply here as well. In particular, your MSE plot would use a legend showing the sparsity of each line. Just looking at it, I can't tell what the different is between the various colored lines. 

This plot also doesn't seem right to me. The MSE is always increasing at every level of sparsity. For sparse datasets, the MSE should decrease with increasing lambda. I think this may be a problem with the way you're defining your sparse vector. It seems like it ought to work just looking at it, but I found an easy way to define a vector with varying levels of sparsity in scipy you could use instead to compare:

```python 
scipy.sparse.random(n, 1, density=0.25)
```

I can't find any of your solutions for the lasso.
