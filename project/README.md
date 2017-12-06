# Final Project

## Background

We now know that the Higgs boson exists, and we know a few things about it, like its mass and spin. So far it behave like we would expect. The obvious next question to ask is, how does it interact with the other particles? My research is specifically stud

The goal, then, is to find a region in phase space which maximizes the significance of the signal with respect to the background. 

## Reading the Data

The Monte Carlo simulations I want to use are stored in ROOT files, so the first step is to convert the information to a form I can use in python. Thankfully, there's a ROOT extension that includes a function for converting ROOT info into CSV format. The script I wrote to do this is [here](create_csv.py). 

```python

```

This generates a file for signal, and a file for each background. A huge number of variables are stored in the ROOT files, so here I'm only saving the features I want to use as input for the BDT. Here's a brief explanation of what these variables are:

* nJets_OR_T : The number of "jets" in the event. Basically, jets are objects that originated from quarks.
* nJets_OR_T_MV2c10_70 : This is the number of jets in the event that look like they came from bottom quarks, or b-jets. Since the top quarks almost always decays to a W boson and a b-quark, we expect to have some of these in ttH. 
* MET_RefFinal_et : This is the amount of missing transverse (perpindicular to the beam) energy in the event. The W boson decays to a lepton and a neutrino, so events with W's are likely to have high MET.
* HT, HT_lep, HT_jets : HT is a measure of the sum of the total momentum in the event. HT_lep and HT_jets would be the sum of the lepton's momentum, and the jet's momentum.
* lead_jetPt, sublead_jetPt : The momentum of the two highest momentum jets in the event.
* best_Z_Mll, best_Z_other_Mll - This is the invariant mass of the two leptons that's closest to the Z boson mass. 
* DRll01, DRll02, DRll12 - This is the seperation between each pair of leptons.
* lep_Pt_0, lep_Pt_1, lep_Pt_2 : This is the momentum of the leptons.

Then, rather than reading in the data from each file every time, I wrote a script that combines each file into a single matrix, X, and saves the result. The background samples are given a vector of zeros, while the 

## Code

## Results

## Conclusion
