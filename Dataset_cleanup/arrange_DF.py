try:
    import feather
except ImportError :
    pass
import numpy as np
import pandas as pd

"""Import clinical (containing data labels) and gene count feather files into
pandas DataFrames"""
try :
    clinical = feather.read_dataframe('feather_files/Clinical_data.feather')
except NameError:
    clinical = pd.read_csv('csv_files/Clinical_data.csv')

try :
    gene_counts = feather.read_dataframe('feather_files/Gene_counts.feather')
except NameError:
    gene_counts = pd.read_csv('csv_files/Gene_counts.csv')

"""Check imports"""
if np.isfinite(clinical.shape[0]) :
    print "Clinical data set imported!\nFeatures:",int(clinical.shape[1]-1),"\nPatients:",clinical.shape[0]
else :
    print "Error in Clinical data set import"

"""Remove features that provide no information"""
def useless_vars(dataset) :
    df = pd.DataFrame(dataset.describe())
    to_drop = df.columns[df.loc['unique'] <= 1]
    print "\n","The following features do not provide any information:","\n",to_drop.values,"\n"
    dataset.drop(to_drop, axis = 1, inplace = True)
    return(dataset)
clinical = useless_vars(clinical)
"""Remove features not known at time of diagnosis"""
def future_vars(dataset) :
    df = pd.DataFrame({'Known_at_diagnosis' : '?'}, index = dataset.columns)
    df.loc[('dateofinitialpathologicdiagnosis','Known_at_diagnosis')] = 'yes'
    df.loc[('daystolastfollowup','Known_at_diagnosis')] = 'no'
    df.loc[('daystodeath','Known_at_diagnosis')] = 'no'
    df.loc[('daystopsa','Known_at_diagnosis')] = 'no'
    df.loc[('gleasonscore','Known_at_diagnosis')] = 'yes'  #this is the point of the biopsy and would typically be known within 2 weeks.
    df.loc[('histologicaltype','Known_at_diagnosis')] = 'no'
    df.loc[('numberoflymphnodes','Known_at_diagnosis')] = 'no'
    df.loc[('pathologyTstage','Known_at_diagnosis')] = 'no'
    df.loc[('psavalue','Known_at_diagnosis')] = 'yes'
    df.loc[('race','Known_at_diagnosis')] = 'yes'
    df.loc[('residualtumor','Known_at_diagnosis')] = 'no'
    df.loc[('radiationtherapy','Known_at_diagnosis')] = 'no'
    df.loc[('vitalstatus','Known_at_diagnosis')] = 'no'
    df.loc[('yearstobirth','Known_at_diagnosis')] = 'yes'
    keep = df[df['Known_at_diagnosis'] != 'no'].index
    dropped = df[df['Known_at_diagnosis'] == 'no'].index
    dataset.drop(dropped.values, axis = 1, inplace = True)
    print "Variables that are not known at initial diagnosis:","\n", dropped.values, "\n"
    print "Variables that are known at the time of diagnosis:\n",keep.values
    return(dataset)
clinical = future_vars(clinical)


if np.isfinite(gene_counts.shape[0]) :
    print "\n\nGene Counts data set imported!\nFeatures:",int(gene_counts.shape[1]-1),"\nPatients:",gene_counts.shape[0]
else :
    print "Error in Gene_counts import"

"""Set index for both data frames as the TCGA ID code"""
clinical.set_index(['clinical_index'], inplace=True) #set index to the TCGA ID
gene_counts.set_index(['gc_index'], inplace = True) # set the index as the TCGA ID codes

"""Retrieve metastasis states from clinical """
y_all = clinical['pathologyNstage'] #pull out the label (metastasis or no metastasis) as y
clinical.drop(['pathologyNstage','dateofinitialpathologicdiagnosis', 'race' ], axis = 1, inplace=True) #drop label from feature set
