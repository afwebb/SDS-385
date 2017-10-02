# Peer Review for Bowen Hua

## General Comments

I went through the files you have in the ex1 folder, and provided comments for each file individually. It is a bit funny, we did take very similar approaches, especially on gradient descent. Which is to say, I like how you approached things. There were actually very few things I can see to improve, just a few places where I think you can streamline your code. I made a few comments about places you're copying and pasting where I'm not sure you need to.

I think it would be helpful if you gave your files more descriptive names. "logit.py" doesn't tell me much about what the program does. That way, if you come back to this code in a few weeks or months, you can have an idea of what each program does, rather than having to look through the code. It would also make things easier for someone else looking at your repository. 

Another thing you might consider is using markdown to display your results. The way you have it, written up in latex, is great, especially when it comes to equations, but I know latex can be a lot of work. Using markdown might save you some time, and it is nice having it show up right when someone goes to your repository.

## [ex1.pdf](https://github.com/bowenhua/SDS_385/blob/master/ex1/ex1.pdf)

I appreciate you took the time to write all this up in latex. It makes it very easy to read, especially the equations, and just makes it look nice. Your descriptions are fairly sparse, but I feel like you've done a good job of making the main points of what you're trying to do clear.

In the section about the pseudoinerse method, I'd have liked if you went into more detail in this section about why this method is useful. You mention numberical stability, but I'd be interested to know what about it makes it more numerically stable than direct inversion.

Including the complexity of each algorithm in big O notation is a nice touch.

For 1.3, your table shows method 2 performing much more slowly than the others. It would be helpful if you gave an explanation for why you think that is. Also, you state that for square matrices, method 3 performs worse, but I'm not I see that looking at the table.

In 1.4, your plot showing the time to complete the solution as a function of sparsity is great. Really illistrates the point you're trying to make clearly.

In problem 2, you go into a good amount detail with your step-by-step solutions (you even have footnotes), so they're easy to follow.

2.2, I really appreciate the way you include each step size in the same plot, this does a good job of demonstrating the effect of different step sizes. The same goes for all your plots. Using 1/k as a step size is also a clever way to use variable step-size that's easy to implement.

## [linear_system.py](https://github.com/bowenhua/SDS_385/blob/master/ex1/linear_system.py)

Around line 59, you calculate the time it takes to run a function several different times. I think it would be worth defining a function that does this for you, rather than you having to repeat it each time. Here's more or less what I did:

```python
def time_solve(f, X, Y, W):
    start = time.time()
    result = f(X, Y, W)
    return result, time.time()-start_inv
```

This would require some reworking of your code, but if you had to do something similar in the future, I think it would save you some time if you based it around something like this. And I think you could do something similar then you're running over several different densities.   

Your comments here are fairly sparse. While each function is labeled, a brief description of what each one does would be nice.

The way you've used several different sizes and shapes is great, and provides a good test of each of the methods you're using.

## [Logit.py](https://github.com/bowenhua/SDS_385/blob/master/ex1/logit.py)

Overall, your code is well written and easy to follow. I appreciate that you have a consistent style for the comments of each function. Its well commented, though you could add a few more. There are a few things you explain in your write up, like why you add a column of 1s to your x-matrix, that aren't clear reading through your code. A few short comments explaining why you're doing certain things would make your code easier to follow.

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

