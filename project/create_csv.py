# Convert root files into .csv files

import rootpy.io
from rootpy.tree import Tree

# Define the variables we want to use
branch_list = ['nJets_OR_T', 'nJets_OR_T_MV2c10_70', 'MET_RefFinal_et', 'HT', 'HT_lep', "HT_jets", 'lead_jetPt', 'sublead_jetPt', 'best_Z_Mll', 'best_Z_other_Mll', 'DRll01', 'DRll02', 'DRll12', 'lep_Pt_0', 'lep_Pt_1', 'lep_Pt_2']

# List the data set IDs to use
dsids = ['363491', '343365', '410155', '410218', '410219']

for dsid in dsids:
    print dsid
    f = rootpy.io.root_open('root_files/'+dsid+'.root')

    # Close all the variable branches. Open the ones we actually want
    oldTree = f.get('nominal')
    oldTree.SetBranchStatus("*",0)
    for br in branch_list:
        oldTree.SetBranchStatus(br,1)

    # Create a new ROOT file with only the necessary branches. Copy all the events from the old file
    newFile = rootpy.io.root_open('root_files/small_'+dsid+'.root', 'recreate')
    newTree = oldTree.CloneTree(0)
    for br in branch_list:
        newTree.GetBranch(br).SetFile('root_files/small_'+dsid+'.root')
        newTree.CopyEntries(oldTree)

    newFile.Write()
    newFile.Close()

    g = rootpy.io.root_open('root_files/small_'+dsid+'.root')
    gTree = g.get('nominal')

    # Write the information from the new file to a csv
    gTree.csv(stream=open('root_files/'+dsid+'.csv', 'w'))

