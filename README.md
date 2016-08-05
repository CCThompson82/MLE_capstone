# MLE_capstone - Prediction of Prostate Cancer Metastasis

This repository contains all the necessary files, scripts, and documentation to predict the probability of prostate cancer metastasis from a malignant biopsy RNA-seq profile (in Transcripts per million gene count format).  This is an exercise for completion of my Machine Learning Engineer Nanodegree from Udacity and should not be treated as a medical or diagnositic tool. 

The project report is titled [Capstone_report.pdf](https://github.com/CCThompson82/MLE_capstone/blob/master/Capstone_report.pdf)

# Dependencies
Execution of the ipython notebook analysis code depends on standard modules:
* pandas
* numpy
* sklearn
* matplotlib

Data set upload into the python environment is enhanced by the [feather](https://github.com/wesm/feather) module, but the module is not necessary.  

Conda install from MacOS command line
`conda install -c jjhelmus feather-format=0.1.0`

Pip install from MacOS command line
`pip install feather-format`

# Data sets
The data used in this project is publically available from [The Cancer Genome Atlas](http://cancergenome.nih.gov/) (TCGA) portal.  It was retrieved and organized with help of an R package, [TCGA2STAT](https://cran.r-project.org/web/packages/TCGA2STAT/index.html). The data sets deposited in this repository were current as of the time of project submission (04-Aug-2016).   

# Implementation
The ipython nb code `PC_capstone.ipynb` is set up to retrieve the primary data sets and support scripts for full execution as long as repository architecture remains the same.  The data sets the notebook will upload were current at the time of project submission, however if the most up to date TCGA prostate cancer is desired, the R script for re-downloading, organizing, and writing the data file locally is available [here](https://github.com/CCThompson82/MLE_capstone/tree/master/Dataset_setup).

# Collaboration
This project is ongoing and welcomes any potential collaborators.  Those capable of helping to implement a pipeline from raw RNA-seq reads to TPM format would be especially welcome.  
