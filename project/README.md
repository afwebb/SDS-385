# Final Project

The goal of my project is apply the techniques we've developed in class to a set of particle physics data. I then do the same with multivariate classification techniques, and compare the different approaches, discussing their advantages and disadvantages.

## Background

My research has to do with studying the properties of the Higgs Boson. While we know it exists, there's good reason to believe studying the Higgs and the way it interacts with other particles could tell us something about Dark Matter, or Super Symmetry, or the [Hierarchy Problem](https://en.wikipedia.org/wiki/Hierarchy_problem#The_Higgs_mass).

The role of the Higgs Boson is to give particles their mass. This makes its interactions with the Top Quark is a good candidate for study: The Top Quark the heaviest fundamental particle, and therefore has the strongest interaction with the Higgs. My research involves analysing data produced by the LHC (Large Hadron Collider) to study the interactions of the Higgs boson and the Top Quark. The interaction we're looking for looks something like this:

<img src="https://github.com/afwebb/SDS-385/blob/master/project/plots/ttH_ML.png" width="300">

Two gluons from the collided protons iteract, which produces a Higgs and Top Quark pairs. The Top Quarks almost always decay to a b-quark and a W-boson, and events where the Higgs decays to two W's are easier to study than some other channels. This means means we want to look for events with two b-quarks, which show up in the detector as "b-jets". W bosons decay to either 2 quarks or a lepton and a neutrino. I focused on events with three leptons, so one W decayed to quarks, the rest to leptons. So what we look for in the detector is events with 4-jets, two of which are b-jets, and three leptons.    

The problem is there are a few other ways to get that final state. Here's one example:

<img src="https://github.com/afwebb/SDS-385/blob/master/project/plots/wz_3l.PNG" width="300">

For this project I'll be trying to distinguish the signal Higgs events from the most common backgrounds using techniques I learned this semester, and a few others from scikit learn, in order to compare their performance on the dataset.

## The dataset

Since we can't tell from data what signal events and background events look like, we simulate them instead. Monte Carlo techniques are used to simulate the physics interactions we're interested in and how those interactions would end up looking in the detector, and we compare these simulations to the data as a way of comparing theory and experiment. More relevant here though, we can ask our simulations which events are signal, and which are background, and therefore build a classifier based on the simulations. 

The Monte Carlo simulations I want to use are stored in ROOT files, so the first step is to convert the information to a form I can use in python. The script I wrote to do this is [here](create_csv.py). This generates a file for signal, and a file for each background. A huge number of variables are stored in the ROOT files, so here I'm only saving the features I want to use as input for the BDT. 

Choosing the feature variables to include is fairly challenging, and while prior knowledge about the physics involved might give you hints about which variables will give you seperation between signal and background, I'm often surprised by which features end up working best. So, I plotted all of the variables I thought might be useful and chose from there.

<img src="https://github.com/afwebb/SDS-385/blob/master/project/plots/variables_id_c3.png" width="600">

Its not worth getting into exactly what all these variables represent, but here's what a few of them are:

* nJets : The number of "jets" in the event. Basically, jets are objects that originated from quarks. 
* MET : This is the amount of missing transverse (perpindicular to the beam) energy in the event. The W boson decays to a lepton and a neutrino, so events with W's are likely to have high MET.
* HT : a measure of the sum of the total momentum in the event. 
* DRll01 - This is the seperation between two of the leptons.
* lep_Pt_0: This is the momentum of the leptons.
* lead_jet_Pt: The momentum of the highest energy jet.

Then, rather than reading in the data from each file every time, I wrote a script that combines each file into a single matrix, X, and saves the result, see [here](read_data.py). I write 100,000 events to the file, since adding more didn't give me any better results.

There are a few complications I'm not including here. For one, I'm training on an equal amount of signal and background, when in reality, the signal is about ~10 times less common than background. I did try to do classification on a proportionate amount of signal and background, but the classifiers ended up doing almost no better than you would be assuming everything was background. For another, I'm trusting the Monte Carlo to exactly model my data. For an actual paper, a lot of work would go into validating the simulations, and taking into account their inherent uncertainties.

I'm ingnoring these things because I'm more interested in how well these techniques work on this kind of data, and which ones work best.

### Gradient Descent

I started with trying to do the best linear classification I could for the problem. I ended up choosing gradient descent over, for example, the lasso, since that was designed to exploit sparsity and my data is fairly dense. The feature I've selected has already winnowed out most of the irrevelent features, so doing full gradient descent with a penalty term seemed like the best approach.

[Here](gradient_descent.py) is the code for this part. Using full gradient descent, the loss function converged after about 170 iterations. This took a long time to complete, and did an okay job of accurately predicting whether an event was signal or background:

```python
Time to complete: 423.82
Accuracey of prediction: 0.88
```

<img src="https://github.com/afwebb/SDS-385/blob/master/project/plots/result_gradient.png" width="500">

Using stochastic gradient (code located [here](sgd.py) ) converged far more quickly, and acheived a slightly better result, surprisingly, than gradient descent did. 

``` python
Time to complete: 4.09
Accuracey of prediction: 0.89
```

<img src="https://github.com/afwebb/SDS-385/blob/master/project/plots/result_sgd.png" width="500">

This actually acheived a better result than I expected it to; a ninety percent accuracey is quite good, considering how similar background and signal events are. The variable distribution plots I included above show very similar shapes, which made me doubt how useful a linear classifier would be.

### BDT

The topics we discussed in class were mainly focused on linear regression, but many non-linear classifiers are based on the same fundational principles. Gradient Boosted Decisions Trees are one example: As with linear regression, the gradient is used to find a minimum of a convex loss function. It is a general use function that can be applied to many different types of classification problems. 

The idea is to start with a simple decision tree which divides events into two categories based on a single feature, one more signal like, the other more background like. The weight of each split is determined using the line search method discussed in the week 3 exercises. Each split is determined by maximizing the gradient of the loss function with respect to the feature set. Additional decision trees are added to correct the mistakes of the model so far. This model can take a long time to generate, but once it has, a data set can quickly be run through the set of decision trees to give a classification.

<img src="https://github.com/afwebb/SDS-385/blob/master/project/plots/tree_example.png" width="300">

This basic algorithm can be improve by introducing an l2 penalty term, that discourages the production of complex trees. This helps prevent overfitting the training data. Of course, cross validation is generally applied as well. Another common improvement is to introduce a "learning rate". Each tree that is added to model is supressed by a scalar factor, typically order 0.1, in order to stabalize the algorithm. While the learning rate slows the computation time, it has empirically been shown to improve results. 

One advantage of BDTs is that they require few input parameters on the part of the user. For example, the model I used in scikit learn only required I specify the learning rate, the max depth of the trees, and the number of trees. This makes BDTs relatively stable, and helps prevent some of the pitfalls that can come with too much fine tuning.

The code I used to produce run my BDT can be found [here](bdt.py). The algorithm assigns each event a total score, based on where it falls in each of the decision trees.  The plot below each event a total score, based on where it ends up at the bottom of each tree. I plotted this output for signal and background events, which gives an idea of how well the algorithm is doing at seperating signal from background:

<img src="https://github.com/afwebb/SDS-385/blob/master/project/plots/bdt_dis.png" width="500">

``` python 
Time to complete: 146.4
Accuracey: 0.96
```
This is a pretty significant improvement over linear regression. Of course, it took quite a lot longer than stochastic gradient descent to get there.

I ended up using 250 trees and a learning rate of 0.1 to get this result. This seemed to give the best results; adding more trees didn't help. I plotted the error as a function of the number of trees in the model, and it suggests things converged nicely:

<img src="https://github.com/afwebb/SDS-385/blob/master/project/plots/bdt_error.png" width="500">

This seems like a fair plot to compare to the shape of the likelihood plot I showed for gradient descent, and the shape even looks surprisingly similar.

### MLP

An MLP, or multi-layer perceptron, is an example of a neural net. Input features of the model are combined and fead into an activation function, typically hyperbolic tangent or a sigmoid. These activation functions serve as a hidden layer of the neural net. The output of the hidden layer can be fead into another hidden layer, before finally producing a classification output.

<img src="https://github.com/afwebb/SDS-385/blob/master/project/plots/MLP.jpg" width="400">

Neural nets generally, and MLPs specifically, tend to be well suited for finding complex patterns in data, making them using for things like image recognition. Most people in particle physics use BDTs most of the time, but I have seen a few instances where neural nets end up giving better results. 

My mlp code can be found [here](mlp.py). I discovered an option to add momentum to the algroithm, and ran it with a constant learning rate, as well as momentum and Nesterov momentum. I got the following result:

<img src="https://github.com/afwebb/SDS-385/blob/master/project/plots/mlp_result.png" width="500">

```python
training: constant learning-rate
Time to complete: 104.886443853
Training set score: 0.950080
training: with momentum
Time to complete: 29.8559989929
Training set score: 0.951400
training: with Nesterov's momentum
Time to complete: 35.6580569744
Training set score: 0.953730
```

Momentum clearly makes a big difference, and surprisingly even gives better results. The results here are comparable to using a BDT, if slightly worse. 

## Conclusion

While multivariate techniques generally outperform the linear ones, there are quite a few reasons one might not want to avoid using a non-linear classifier. 

One problem is that the final result we get after training on a BDT for example, obscures some of the physics behind that result. Saying we observed a certain number of events in this complex region defined by the BDT limits the number of conclusions you can draw from your analysis. For trying to discover the Higgs, for example, this might be okay: You could conclude that the number of events in this complex BDT output region is consistent with data only if the Higgs Boson exists. If, on the other hand, you're trying to measure something precisely, a linear approach is probably better. With a linear system, have a well defined region of space (e.g. events with 3 leptons whose energy is above a certain value determined by the classifier) which you can use to extrapolate, for example, how often you expect a particular interaction to occur. 

Another issue that needs to be considered is systematic uncertainties. The effect of things like the resolution of the detector, the theoretical limitations of our predictions, are hard to disintangle with a nonlinear classifier. If a particular systematic uncertainty has a large impact on our result, it can be difficult to see where in the classifier this systematic is having an affect.  

That said, my results suggest that these nonlinear algorithms do improve the results you get, though not as I might hve expected. 
These results suggest to me that using an MVA can be valuable, but the improvement you acheive form using them needs to be carefully weighed against some of the downsides.
