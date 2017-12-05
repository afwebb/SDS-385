#Convert root files into .csv files

import rootpy.io
from rootpy.tree import Tree

branch_list = ['nJets_OR_T', 'nJets_OR_T_MV2c10_70', 'MET_RefFinal_et', 'HT', 'HT_lep', "HT_jets", 'lead_jetPt', 'sublead_jetPt', 'best_Z_Mll', 'best_Z_other_Mll', 'DRll01', 'DRll02', 'DRll12', 'lep_Pt_0', 'lep_Pt_1', 'lep_Pt_2']
dsids = ['363491', '343365', '410155', '410218', '410219']

for dsid in dsids:
    print dsid
    f = rootpy.io.root_open('/data_ceph/afwebb/gfw2/output/'+dsid+'.root')
    oldTree = f.get('nominal')
    oldTree.SetBranchStatus("*",0)
    for br in branch_list:
        oldTree.SetBranchStatus(br,1)

    newFile = rootpy.io.root_open('small_'+dsid+'.root', 'recreate')
    newTree = oldTree.CloneTree(0)
    for br in branch_list:
        newTree.GetBranch(br).SetFile('small_'+dsid+'.root')
        newTree.CopyEntries(oldTree)

    newFile.Write()
    newFile.Close()

    g = rootpy.io.root_open('small_'+dsid+'.root')
    gTree = g.get('nominal')

    gTree.csv(stream=open(dsid+'.csv', 'w'))

