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


