# Peer Review for Bowen Hua

## [ex1.pdf](https://github.com/bowenhua/SDS_385/blob/master/ex1/ex1.pdf)

I appreciate you took the time to write all this up in latex.

## [linear_system.py](https://github.com/bowenhua/SDS_385/blob/master/ex1/linear_system.py)

Around line 59, you calculate the time it takes to run a function several different times. I think it would be worth defining a function that does this for you, rather than you having to repeat it each time. Here's more or less what I did:

```python
def time_solve(f, X, Y, W):
    start = time.time()
    result = f(X, Y, W)
    return result, time.time()-start_inv
```

This would require some reworking of your code, but if you had to do something similar in the future, I think it would save you some time if you based it around something like this.    

Your comments here are fairly sparse. While each function is labeled, a brief description of what each one does would be nice.

## [Logit.py](https://github.com/bowenhua/SDS_385/blob/master/ex1/logit.py)
Overall, your code is well written and easy to follow. Its well commented, though you could add a few more. 

When running the code, I think you could streamline things by looping over several step sizes. Rather than doing this:

```python
beta = np.random.rand(11,1) 
beta1 = beta.copy()
beta2 = beta.copy()
beta3 = beta.copy()
beta4 = beta.copy()
beta5 = beta.copy()

beta1, costs1 = optimize(beta1, trainX, trainY, 50, '0.001')
beta2, costs2 = optimize(beta2, trainX, trainY, 50, '0.005')
beta3, costs3 = optimize(beta3, trainX, trainY, 50, '0.01')
beta4, costs4 = optimize(beta4, trainX, trainY, 50, '1/k')
beta5, costs5 = optimize(beta5, trainX, trainY, 50, 'newton')
```
Something like this is how I did it:

```python
beta_original = np.random.rand(11,1)
iterations = 50

for step_size in [0.001, 0.005, 0.01, 1/k]:
  beta = beta_original.copy()
  beta, cost = optimize(beta1, trainX, trainY, iterations, step_size)
```

Which should have the same functionality with fewer lines of code. I'd also recommend having the number of iterations be a variable, rather than hard coded for each run. 

I know for me, when I copy and paste, it ends up causing me problems down the road. I'll want to change one parameter, and end up forgetting to change it everywhere. By running in a loop like this, and by setting as many parameters to variables as possible, it makes it easier to make changes to your code, or add on to it in the future. 

I'd recommend streamlining your optimize() function in a similar way. Here's how you have it now:

```python
    if step_size == '1/k':
        for i in range(num_iterations):      
            dbeta, cost = propagate(beta, X, y) 
            if max(abs(dbeta))<1e-7:
                break
            beta -= dbeta * (1/(num_iterations))  
            costs.append(cost.flatten())
    elif step_size == 'newton':
        for i in range(num_iterations):
            dbeta, cost = propagate(beta, X, y)
            if max(abs(dbeta))<1e-7:
                break
            delta_beta = np.linalg.solve(hessian(beta, X),dbeta)            
            beta -= delta_beta  
            costs.append(cost.flatten())
    else: # constant step size
        step_size = float(step_size)
        for i in range(num_iterations):
            dbeta, cost = propagate(beta, X, y)  
            if max(abs(dbeta))<1e-7:
                break
            beta -= dbeta * step_size  
            costs.append(cost.flatten())
```

There's only one line of code that's different between these three cases. I think it would be much simpler if, rather than defining the entire function for each case, you minimized the amount of code you had to repeat:

```python
 for i in range(num_iterations):      
    dbeta, cost = propagate(beta, X, y) 
    if max(abs(dbeta))<1e-7:
       break
    if step_size == '1/k':
       beta -= dbeta * (1/(num_iterations)) 
    elif step_size == 'newton':
       delta_beta = np.linalg.solve(hessian(beta, X),dbeta)            
       eta -= delta_beta
    else:
       beta -= dbeta * step_size
    costs.append(cost.flatten())
```

I do like that you managed to have newton's method run within the same function as the rest. You also came up with a really elegant way to test the convergence of the loss function. Your plot looks really good, especially how you fit the results for each of the different step sizes you used onto one graph. I also like how you tested the accuracey of the prediction, and you managed to do it in a very simple way.

## General Comments

I think it would be helpful if you gave your files more descriptive names. That way, if you come back to this code in a few weeks or months, you can have an idea of what each program does, rather than having to look through the code.
