# Final Project

## Background

The Higgs is interesting because particles get their mass by interacting with it. And while we know it exists, there's good reason to believe studying the Higgs and the way it interacts with other particles could tell us something about Dark Matter, or Super Symmetry, or the [Hierarchy Problem](https://en.wikipedia.org/wiki/Hierarchy_problem#The_Higgs_mass).

The Top Quark is an obvious place to look: It's the heaviest fundamental particle, and therefore has the strongest interaction with the Higgs. My research involves analysing data produced by the LHC (Large Hadron Collider) to study the interactions of the Higgs boson and the Top Quark.

This means means we want to look for events with a Higgs Boson, and a pair of Top Quarks. For this project I'll be trying to distinguish these events from backgrounds using techniques I learned this semester, and a few others from scikit learn, in order to compare their performance on the dataset. 

I adapted some of the techniques we learned in class to work with my dataset, namely gradient descent, scochastic gradient descent, and ADMM. I'm also using Boosted Decision Trees and Multi Layer Perceptrons from scikit-learn.

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

<img src="https://github.com/afwebb/SDS-385/blob/master/project/result_gradient.png" width="500">

```python
Time to complete: 323.824832916
Accuracey of prediction: 0.7443308
Significance of signal: 357.529370346
```

### Stochastic Gradient Descent

<img src="https://github.com/afwebb/SDS-385/blob/master/project/result_sgd.png" width="500">

### Proximal Gradient Descent

Just like the inclass exercises, I ran both with and without momentum.

### ADMM 

### BDT

### MLP

## Conclusion
