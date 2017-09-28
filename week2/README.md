# Excercises 2

## SGD for Logistic Regression

The code for this excercise can be found [here](stocastic.py)

The procedure is similar to the previous excercise, except the slope of the loss function is taken for a random point in the dataset. The random sampling may not give the exact slope at each step, but it will minimize the function over the course of a large number of iterations. 

The program runs through each point in the dataset, then shuffles the entries and runs through again. It continues to run until a convergence criteria is met. 

```python
while not test_converge(l, e):
        j = i%len(y)
        if j==0:
            X, y = shuffle(X, y)
        b -= calc_gradient(X, y, b, j)*step_size
        l.append(calc_l(X, y, b))
        i+=1

        if i>1000000:
            break
```

<img src="https://github.com/afwebb/SDS-385/blob/master/week2/result_0.01.png" width="500">

```python
Accuracey: 0.90
```

<img src="https://github.com/afwebb/SDS-385/blob/master/week2/result_0.1.png" width="500">

```python
Accuracey: 0.93
```

<img src="https://github.com/afwebb/SDS-385/blob/master/week2/result_1.png" width="500">

```python
Accuracey: 0.93
```

This results are comparable to using full gradient descent, but with much less computation. I set the convergense requirement to 0.01% change in likliehood, and still for each step size, the data set is ran over less than twice before the algorithm converges.




