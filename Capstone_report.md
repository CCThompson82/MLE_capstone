# Definition

## Project Overview

The prostate is a glandular organ of the male reproductive system that helps to
control urinary and reproductive functions.  According to the charity Prostate
Cancer UK, one in eight British men will  be diagnosed with prostate
adenocarcinoma (prostate cancer, henceforth 'PC') in their  lifetime {citation}.
Men over 50 years of age are often subjected to routine digital examinations, or
urine test (the 'PCA' test) for signs of PC.  However the gold standard
diagnosis is the Gleason test.  In brief, a series of small needle sized
biopsies are taken from the patient's prostate gland.  Each biopsy is processed
and scored by a pathologist for signs of abnormal cell type and
structure.  Gleason scores ranging from 2 to 5 are considered not
malignant, whereas scores ranging from 6-10 are considered malignant and provide
an added estimation of severity (Humphrey, 2004).

Contrary to some types of cancer, malignancies that remain local within the
prostate are rarely lethal (survival rate of ~99%) {citation}.  However, if a
malignancy born of the prostate undergoes metastasis (the process of cancer cell
migration to other sites in the body), the survival rate drops to ~28%.  Because
of this discrepancy, many men opt for radical prostatectomy (surgical removal of
the entire prostate).  While ensuring prevention of metastasis, removal of the
prostate results in high morbidity, e.g. inability to control urination, loss of
sexual function, etc.

Unfortunately, there are currently no prognostic tests for PC metastasis.  The
patient data  that is typically available at the time of diagnosis is not rich
enough to accurately predict the likelihood of prostate cancer metastasis
{citation}.   A model that is able to predict whether an untreated malignancy
would be likely to remain locally within the prostate or to metastasize could be
an invaluable tool on whether prostatectomy (and the associated morbidity) is
necessary.  To generate such a model, a more distinguishing set of feature data
is required.

One potential solution to this problem is an RNA-seq profile.  In brief, RNA-seq
is a technique that reads and counts RNA sequences in a biological specimen
sample.  What is RNA?   When a gene is activated in a cell, the DNA sequence is
read ('transcribed') into an RNA molecule.  Thus by reading all of the RNA
molecules that exist in a sample, one may determine which genes have been
activated, and to what degree.  A gene count profile is the estimation of
activation for the full set of known human genes.

As metastatic cancer cells behave in drastically different ways than malignant
cells that remain local, there must be an inherent difference in gene activation
between the two cell types.   This difference may be detectable, however there
are likely to be many different genetic paths towards metastasis. Thus it is
unlikely that a single gene could distinguish metastasis state and local
malignancy. The ultimate goal of this project is to determine whether the
RNA-seq profile taken from a cancerous prostate biopsy during initial
presentation and diagnosis, is  sufficient for prognosis of prostate cancer
metastasis.

## Problem Statement

The primary questions that this project aims to answer are :

* Can the risk of prostate cancer metastasis state be predicted from a gene activation (RNA-seq) profile?

* If so, what genes (individually or in concert) are important for this assessment?

The goal of this project is to design a model that predicts the risk of prostate
cancer metastasis using the gene activation profile derived from the patient's
prostate biopsy, taken at the initial Gleason Test diagnosis phase.

To achieve this goal, it is likely that a  significant feature reduction
exercise will be necessary, as each  RNA-seq profile quantifies expression of
20501 human genes.  After feature reduction, a model will be generated to
quantify the risk of prostate cancer metastasis (probability from 0 to 1).
Finally, a function or application will be engineered that receives an RNA-seq
profile as an input and outputs a prediction for future metastasis.

## Metrics

An appropriate metric for the assessment of class probability prediction is the
Logarithmic Loss (Log Loss) score.   

The equation for log loss is :
$$ logloss = -\frac{1}{N} \sum_{i=1}^{N} (y_i *log(p_i) + (1- y_i) * log(1-p_i)) $$
, where _p_ represents an observations
predicted probability (0 < _p_ < 1) and _y_ represents the actual binomial class
({0,1}).

The log loss function provides a penalty score for each predicted observation in
relation to the difference between the actual class {0,1} and predicted
probability (0:1).  Predictions that are both incorrect and confident are
punished harshly.  For instance, were a prediction to call an example as
absolutely true (1) and the example was actually false (0), then infinity would
be returned. Thus, the log loss function in sklearn caps predicted probabilities
away from 0 or 1 to  avoid this situation.  An ultra-conservative model that
predicted 0.5 for every observation (effectively not taking either stance in
classification) would have a benchmark log loss score of approximately 0.693147.

# Analysis

## Data Exploration

'The Cancer Genome Atlas'(TCGA) is a research consortium set up to curate
clinical data from thousands of patient participants, covering an array of
cancer types.   Provided data includes basic clinical information as well as DNA
and RNA sequencing of cancer biopsies. These data sets are updated frequently as
new information becomes available.  Thus each data download represents a
snapshot in an evolving data set.

While detailed genomic and RNA sequence data is control-accessed, pre-processed
gene count data is publicly available.  Data can be downloaded via the
consortium  portal or acquired into data frame format using a package in the R
language.  The versions stored in the submission repository are current at
the time of submission.

The clinical data set contains 22 features, of which several are irrelevant
(e.g. all prostate cancer patients are 'male').  Of the features, three would be
known at or very near the time of presentation: age, PSA test score, and Gleason
score.  

The primary data set to be used in this project is the gene count matrix.  This
data set provides a value for gene expression level for every known human gene.
The same patient index links the clinical data set to the  gene count data set,
of which 497 are common among the two.

The outcome variable for this project is contained in the clinical data set,
which is 'pathologyNstage'.  This label is composed of 'n0' or 'n1',
representing local versus metastatic cancer, respectively.  The current
percentage of metastatic cases is approximately 16%.

![Figure 1](/Figures/Label_count.png)

**Figure 1** - Frequency of Metastasis state ('pathologyNstage') in the TCGA Prostate adenocarcinoma cohort.

The primary data set for this project is the gene count matrix.  This data set
provides a value for gene expression level for every known human gene.  The same
patient index links the clinical data set to the  gene count data set, of which
497 are common among the two.  As the expression values for each patient are not
normalized, an important step prior to analysis was to transform each value to
the transcript count per million reads (TPM).  This transformation normalizes
such that the expression levels for each patient are now comparable.  A test was
run after transformation to ensure that each patient profile totaled 1 million
reads.  The next section will serve to describe the clinical data and provide a
benchmark prognosis rate using information available to a doctor at
presentation.

When grouped by Gleason score, it is also evident that
metastasis rates increase as the severity of cancer increases (Figure 2).  This
is  intuitive, yet clearly not sufficient to determine whether a specific
cancer, regardless of Gleason score, will metastasize or not.  To illustrate,
cancers that have been rated at a Gleason score of '9' are still more likely to
belong to the 'n0' class than the metastasis class.

![Figure 2](/Figures/Gleason_hist.png)

**Figure 2** - Frequency of Metastasis state grouped by Gleason score.  

## Exploratory Visualization

![Figure 3](/Figures/clin_scatter_matrix.png)

**Figure 3** - Distribution of known clinical features grouped by metastasis state (blue: 'n0', red: 'n1').

An overview analysis shown in Figure 4 of the gene count dataset revealed that
some genes appear to be differentially activated in the two metastasis states.
This indicates that there are genes that could be used for predictive purposes
and validates the project rationale.

![Figure 4](/Figures/F_distribution.png)

**Figure 4** - F-test statistic distribution for the comparison of gene expression levels between the 'n0' and 'n1' metastasis states.  


## Algorithms and Techniques

The basic outline for project completion is as follows:

1. Feature reduction (filter)
2. Feature compression into a lower dimensional set (removes collinearity)
3. Training of the probabilistic-classification algorithm
4. Measure performance of the trained algorithm on an unseen 'test' set, and compare to the benchmark model performance.  

The feature reduction exercise will utilize Random Forest Classifier, not as a
classification algorithm, but as a method to measure the ability of each gene to
separate the dataset by metastasis class.  Given noisy data, decision trees (and
thus Random Forest) classifiers are prone to overfitting, so parameter limits on
the tree depth and the minimum number of samples that can be split will be defined.  
The top portion of genes will be subset into a new reduced data set and carried
into the next project phase.  

The reduced feature data set will be compressed further using Principle Component
Analysis (PCA).  PCA is an unsupervised learning technique that transforms a
dataset into its principle components - _i.e._ the orthogonal vectors within the
data that explain the greatest amount of its variance.  By selecting the
the most important components, several features may be combined into a lower
number without significant loss of information.  How many principle components
will be carried into the algorithm training will depend on the amount of variance
each component can explain.  For example, if the first principle component that
explains 95% of the dataset variance, it would not be necessary to bring any other
principle components forward for further analysis.  

The probabilistic-classification algorithm chosen for this task is the logistic
regression ('logit') model.  This algorithm was chosen for its inherent ability
to assess the probability of a binary outcome (_i.e._ metastasis or local
malignancy) based on continuous input variables.  

Important parameters for the Logistic Regression are:

* regularization equation ('penalty')  
* solver
* regularization term ('C')



## Benchmark

As personalized medicine (_e.g._ use of a patient's specific genetic or gene
activation information for therapeutic decisions) has not been established in
mainstream therapy, a benchmark for use of RNA-seq data for prognosis of
metastasis was not available.  Hypothetically, the most conservative model which
predicts every test sample as having 50% chance of metastasis would yield a log
loss score of ~0.69314.   

To establish a more fair benchmark for comparison, a logistic regression model
that incorporated the clinical information that would normally be known at the
time of diagnosis was generated.  These features were 'age', 'PSA score', and
'Gleason score'.

![Figure 5](/Figures/benchmark.png)

**Figure 5** - Visualization of a benchmark logistic regression predictive model performance.  

The coefficients for the three features in the model training exhibited that
Gleason score was by far the most predictive (0.855), and that age and
interestingly PSA score (which is the current default test that doctors rely on
for prostate cancer risk)  provided very little use in classification.  Figure 5
(left) shows the relationship between Gleason score and the benchmark model's
prediction of metastasis.  Figure 5 (right) shows the distribution of metastasis
probabilities, grouped by actual metastasis state.

The log loss score from this benchmark analysis generally ranged from 0.55 - 0.65, and
thus was marginally more useful than a '50% model'.    

# Methodology

##Data Preprocessing

During the course of the exploratory visualization, it was apparent that no
specimens with a Gleason score of 6 were also metastatic (Figure 2).  Notably,
the metastasis state for many cases in this Gleason category were missing.  In
order to make more efficient use the TCGA cohort data set, these cases with
missing metastasis state were assumed to be 'n0'.  The justification for this
decision lies in the fact that a doctor would likely not require a metastasis
test for mild cases, and thus a non-metastatic cancer should be assumed in these
cases.  Given a much bigger data set, then this assumption would not be
necessary, and  all cases with missing label could be excluded.  Indeed, for
cases scored Gleason  7-10 with missing metastasis labels were excluded from
analysis.  

The primary gene count data set was normalized in such that each patient's gene
count profile summated to 1 million reads (_i.e._ each value is in transcript
per million reads, or TPM, format).  Patient samples in this data set were
limited to those with a metastasis label ('X' - 446 samples by 20501 gene
features).  This DataFrame was then used to train a Random Forest Classifier.  
The point of this exercise was not classification, but to assess the importance
for each gene in class separation utility.  The top 20 genes from this analysis
were retained.  
