# Final Project

The goal of my project is apply the techniques we've developed in class to a set of particle physics data. I then do the same with multivariate classification techniques, and compare the different approaches.

## Background

My research has to do with studying the properties of the Higgs Boson. While we know it exists, there's good reason to believe studying the Higgs and the way it interacts with other particles could tell us something about Dark Matter, or Super Symmetry, or the [Hierarchy Problem](https://en.wikipedia.org/wiki/Hierarchy_problem#The_Higgs_mass).

The role of the Higgs Boson is to give particles their mass. This makes its interactions with the Top Quark is a good candidate for study: The Top Quark the heaviest fundamental particle, and therefore has the strongest interaction with the Higgs. My research involves analysing data produced by the LHC (Large Hadron Collider) to study the interactions of the Higgs boson and the Top Quark. The interaction we're looking for looks something like this:

<img src="https://github.com/afwebb/SDS-385/blob/master/project/plots/ttH_ML.png" width="500">

Two gluons from the collided protons iteract, which produces a Higgs and Top Quark pairs. The Top Quarks almost always decay to a b-quark and a W-boson, and events where the Higgs decays to two W's are easier to study than some other channels. This means means we want to look for events with two b-quarks, which show up in the detector as "b-jets". W bosons decay to either 2 quarks or a lepton and a neutrino. I focused on events with three leptons, so one W decayed to quarks, the rest to leptons. So what we look for in the detector is events with 4-jets, two of which are b-jets, and three leptons.    

The problem is there are a few other ways to get that. For this project I'll be trying to distinguish these events from backgrounds using techniques I learned this semester, and a few others from scikit learn, in order to compare their performance on the dataset. I adapted some of the techniques we learned in class to work with my dataset, namely gradient descent, scochastic gradient descent. I'm also using Boosted Decision Trees and Multi Layer Perceptrons from scikit-learn.

## Reading the Data

The Monte Carlo simulations I want to use are stored in ROOT files, so the first step is to convert the information to a form I can use in python. Thankfully, there's a ROOT extension that includes a function for converting ROOT info into CSV format. The script I wrote to do this is [here](create_csv.py). 

This generates a file for signal, and a file for each background. A huge number of variables are stored in the ROOT files, so here I'm only saving the features I want to use as input for the BDT. Its not worth getting into exactly what all these variables represent, but here's what a few of them represent:

* nJets : The number of "jets" in the event. Basically, jets are objects that originated from quarks. 
* MET : This is the amount of missing transverse (perpindicular to the beam) energy in the event. The W boson decays to a lepton and a neutrino, so events with W's are likely to have high MET.
* HT : a measure of the sum of the total momentum in the event. 
* DRll01, DRll02, DRll12 - This is the seperation between each pair of leptons.
* lep_Pt_0, lep_Pt_1, lep_Pt_2 : This is the momentum of the leptons.

Then, rather than reading in the data from each file every time, I wrote a script that combines each file into a single matrix, X, and saves the result, see [here](read_data.py). The background samples are given a vector of zeros, while the 

### Gradient Descent

<img src="https://github.com/afwebb/SDS-385/blob/master/project/plots/result_gradient.png" width="500">

```python
Time to complete: 323.824832916
Accuracey of prediction: 0.7443308
Significance of signal: 357.529370346
```

### Stochastic Gradient Descent

<img src="https://github.com/afwebb/SDS-385/blob/master/project/plots/result_sgd.png" width="500">

### Proximal Gradient Descent

Just like the inclass exercises, I ran both with and without momentum.

### ADMM 

### BDT

The topics we discussed in class were mainly focused on linear regression, but many non-linear classifiers are based on the same fundational principles. Gradient Boosted Decisions Trees are one example: As with linear regression, the gradient is used to find a minimum of a convex loss function.

<img src="https://github.com/afwebb/SDS-385/blob/master/project/plots/tree_example.png" width="500">

### MLP

## Conclusion
