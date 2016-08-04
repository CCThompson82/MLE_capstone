# MLE_capstone - Prediction of Prostate Cancer Metastasis

This repository contains all the necessary files, scripts, and documentation to predict the probability of prostate cancer metastasis from a malignant biopsy RNA-seq profile (in Transcripts per million gene count format).  This is an exercise for completion of my Machine Learning Engineer Nanodegree from Udacity and should not be treated as a medical or diagnositic tool.  

# Data sets
The data used in this project is publically available from [The Cancer Genome Atlas](http://cancergenome.nih.gov/) (TCGA) portal.  It was retrieved and organized with help of an R package, [TCGA2STAT](https://cran.r-project.org/web/packages/TCGA2STAT/index.html). The data sets deposited in this repository were current as of the time of project submission (04-Aug-2016).   

# Implementation
The easiest way to implement the project code is to clone this repository: https://github.com/CCThompson82/MLE_capstone.git
  and run `Capstone_nb.ipynb`.  This file will retrieve the primary data sets and support scripts / files necessary for its execution.
  If the most current update of the TCGA prostate cancer data sets are desired, the script for re-downloading, organizing, and writing the data file locally is available [here](https://github.com/CCThompson82/MLE_capstone/tree/master/Dataset_setup).

# Collaboration
This project is ongoing and welcomes any potential collaborators.  Those capable of helping to implement a pipeline from raw RNA-seq reads to TPM format would be especially welcome.  
