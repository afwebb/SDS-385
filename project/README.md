# Final Project

The goal of my project is apply the techniques we've developed in class to a set of particle physics data. I then do the same with multivariate classification techniques, and compare the different approaches, discussing their advantages and disadvantages.

## Background

My research has to do with studying the properties of the Higgs Boson. While we know it exists, there's good reason to believe studying the Higgs and the way it interacts with other particles could tell us something about Dark Matter, or Super Symmetry, or the [Hierarchy Problem](https://en.wikipedia.org/wiki/Hierarchy_problem#The_Higgs_mass).

The role of the Higgs Boson is to give particles their mass. This makes its interactions with the Top Quark is a good candidate for study: The Top Quark the heaviest fundamental particle, and therefore has the strongest interaction with the Higgs. My research involves analysing data produced by the LHC (Large Hadron Collider) to study the interactions of the Higgs boson and the Top Quark. The interaction we're looking for looks something like this:

<img src="https://github.com/afwebb/SDS-385/blob/master/project/plots/ttH_ML.png" width="300">

Two gluons from the collided protons iteract, which produces a Higgs and Top Quark pairs. The Top Quarks almost always decay to a b-quark and a W-boson, and events where the Higgs decays to two W's are easier to study than some other channels. This means means we want to look for events with two b-quarks, which show up in the detector as "b-jets". W bosons decay to either 2 quarks or a lepton and a neutrino. I focused on events with three leptons, so one W decayed to quarks, the rest to leptons. So what we look for in the detector is events with 4-jets, two of which are b-jets, and three leptons.    

The problem is there are a few other ways to get that final state. Here's one example:

<img src="https://github.com/afwebb/SDS-385/blob/master/project/plots/wz_3l.PNG" width="300">

For this project I'll be trying to distinguish the signal Higgs events from the most common backgrounds using techniques I learned this semester, and a few others from scikit learn, in order to compare their performance on the dataset. I adapted some of the techniques we learned in class to work with my dataset, namely gradient descent, scochastic gradient descent. I'm also using Boosted Decision Trees and Multi Layer Perceptrons from scikit-learn.

## Reading the Data

Since we can't tell from data what signal events and background events look like, we simulate them instead. Monte Carlo techniques are used to simulate the physics interactions we're interested in and how those interactions would end up looking in the detector, and we compare these simulations to the data as a way of comparing theory and experiment. More relevant here though, we can ask our simulations which events are signal, and which are background, and therefore build a classifier based on the simulations. 

The Monte Carlo simulations I want to use are stored in ROOT files, so the first step is to convert the information to a form I can use in python. The script I wrote to do this is [here](create_csv.py). This generates a file for signal, and a file for each background. A huge number of variables are stored in the ROOT files, so here I'm only saving the features I want to use as input for the BDT. Its not worth getting into exactly what all these variables represent, but here's what a few of them represent:

* nJets : The number of "jets" in the event. Basically, jets are objects that originated from quarks. 
* MET : This is the amount of missing transverse (perpindicular to the beam) energy in the event. The W boson decays to a lepton and a neutrino, so events with W's are likely to have high MET.
* HT : a measure of the sum of the total momentum in the event. 
* DRll01 - This is the seperation between two of the leptons.
* lep_Pt_0: This is the momentum of the leptons.
* lead_jet_Pt: The momentum of the highest energy jet.

Then, rather than reading in the data from each file every time, I wrote a script that combines each file into a single matrix, X, and saves the result, see [here](read_data.py). The background samples are given a vector of zeros, while the 

### Gradient Descent

[Here](gradient_descent.py) is the code for this part. Using full gradient descent, the loss function converged after about 170 iterations. This took a long time to complete, and did an okay job of accurately predicting whether an event was signal or background:

```python
Time to complete: 323.82
Accuracey of prediction: 0.74
```

<img src="https://github.com/afwebb/SDS-385/blob/master/project/plots/result_gradient.png" width="500">



Using stochastic gradient (code located [here](sgd.py) ) converged far more quickly, 

``` python
Time to complete: 4.09
Accuracey of prediction: 0.84
```

<img src="https://github.com/afwebb/SDS-385/blob/master/project/plots/result_sgd.png" width="500">



### BDT

The topics we discussed in class were mainly focused on linear regression, but many non-linear classifiers are based on the same fundational principles. Gradient Boosted Decisions Trees are one example: As with linear regression, the gradient is used to find a minimum of a convex loss function.

The idea is to start with a simple decision tree which divides events into two categories based on a single feature, one more signal like, the other more background like. Each split is determined by maximizing the gradient of the loss function with respect to the feature set. Additional decision trees are added to correct the mistakes of the model so far. This model can take a long time to generate, but once it has, a data set can quickly be run through the set of decision trees to give a classification.

<img src="https://github.com/afwebb/SDS-385/blob/master/project/plots/tree_example.png" width="300">

I used the 

### MLP

<img src="https://github.com/afwebb/SDS-385/blob/master/project/plots/MLP.jpg" width="400">

## Conclusion

While multivariate techniques generally outperform the linear ones, there are quite a few reasons one might not want to avoid using a non-linear classifier. 

One problem is that the final result we get after training on a BDT for example, obscures some of the physics behind that result. Saying we observed a certain number of events in this complex region defined by the BDT limits the number of conclusions you can draw from your analysis. For trying to discover the Higgs, for example, this might be okay: You could conclude that the number of events in this complex BDT output region is consistent with data only if the Higgs Boson exists. If, on the other hand, you're trying to measure something precisely, a linear approach is probably better. With a linear system, have a well defined region of space (e.g. events with 3 leptons whose energy is above a certain value determined by the classifier) which you can use to extrapolate, for example, how often you expect a particular interaction to occur. 

Another issue that needs to be considered is systematic uncertainties. The effect of things like the resolution of the detector, the theoretical limitations of our predictions, are hard to disintangle with a nonlinear classifier. If a particular systematic uncertainty has a large impact on our result, it can be difficult to see where in the classifier this systematic is having an affect.  
