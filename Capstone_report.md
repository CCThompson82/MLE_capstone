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
be returned. Thus, the log loss functions caps predicted probabilities
away from 0 or 1 to  avoid this situation.  An ultra-conservative model that
predicted 0.5 for every observation (effectively not taking either stance in
classification) would have a benchmark log loss score of approximately 0.693147.

# Analysis

## Data Exploration

['The Cancer Genome Atlas' (TCGA)](www.http://cancergenome.nih.gov/) is a research consortium set up to curate
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
(see methodology below) that incorporated the clinical information that would
normally be known at the time of diagnosis was generated.  These features were
'age', 'PSA score', and 'Gleason score'.

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

## Source Files

The datasets were retrieved from the TCGA portal using an R package,
[TCGA2STAT](https://cran.r-project.org/web/packages/TCGA2STAT/index.html), and
written as a [feather](https://github.com/wesm/feather) files to be uploaded
into the python environment.  The R script and feather files are available in
this project's GitHub repository
[MLE_capstone](https://github.com/CCThompson82/MLE_capstone).  All algorithms
were imported from the [scikit-learn](http://scikit-learn.org/stable/) library,
version 0.17.  

## Data Preprocessing

Samples with a Gleason score of 6 were homogenous in metastasis state (all
'n0'). Therefore, any sample of Gleason score equal to 6 where the metastasis
label was missing, 'n0' was imputed for the missing label.  This assumption is grounded
in the  procedures followed by urologist upon Gleason screening - those with low
grade  malignancy are usually not screened from metastasis and this is probably
the  source of missing data labels in the TCGA cohort.  This step allowed more
efficient use of a rather small dataset.  Given a much bigger data set, then
this assumption would not be necessary, and  all cases with missing label could
be excluded.  Indeed, for cases scored 7-10 on the Gleason scale where no labels
were included in the clinical data were excluded from further analysis.  

The gene count data retrieved from the TCGA portal was in an intermediary
format.  While the raw RNA-sequence reads had been processed into gene
activation estimations, each specimens profile required normalization for
cross-comparison.   Therefore, the initial gene count dataset was transformed to
transcripts per million (TPM) format.  This dataframe ('X') was the base upon
which further feature reduction and test train splitting would be performed.  

## Implementation

Feature reduction was completed in two steps.  The first was to utilize the
generation of a [Random Forest Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier) to supply information regarding the
importance of each gene in the separation of metastasis states.  As the Random
Forest model was not intended for actual classification purposes (due to the
small sample size of the dataset), only key default parameters were altered.  
Specifically, the maximum tree depth was limited to 3 nodes, and the minimum
number of samples that could be split was limited to 30.  These parameter choices
were instrumental in preventing any form of extraordinary variance.  The 'Gini
Impurity' of each feature was retrieved from the model and the genes ranked
in the order of importance.  Originally, the top 200 genes were retained for the
input dataset, however this was later reduced to 20 genes (explanation below).

This 20 gene set was scaled to standard mean and unit variance using the sklearn
[Standard Scaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler).  

The second phase of Feature reduction was [Principle Component Analysis](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA)
compression. A further compression was performed in which PCA was utilized to
transform the 20-feature dataset into its principle components.  The contribution
from the 20 genes in each of the first three principle components is visualized
in Figure 6.  

![Figure 6](/Figures/PCA_explained_variance.png)

**Figure 6** - Explained variance and Gene feature contribution to the first three principle components of the PCA transformation.  

This 3-feature dataset was then partitioned using the same indices from the
first
[train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html)
that was performed prior to the benchmark model generation.  In detail, this
split  partioned 70% of the samples into the training set, with 30% being held
out for validation.  The  data was stratified by Gleason score, which was used
as a surrogate measure for cancer severity. While not a perfect solution, this
decision was made to ensure that 'easy' (_e.g._ mild or extremely severe
malignancies) and 'difficult' (_e.g._ malignancies on the border between
moderate and severe) cases would be distributed equally.  Another option would
have been to stratify by  metastasis label (see 'Reflection' section for
discussion on this decision).  

The training data set was then fed to a [Logistic
Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) model.
For this learning, the class-weight parameter was set to 'balanced' in order to
guard against confounding effects of the unbalanced label set in model
performance.  The regularization ('C') parameter was left at the default value
of 1.  The C term is inversely proportional to the penalties awarded for
misclassified samples. As this dataset appears to be noisy from graphical
analysis, I hypothesized a higher regularization term may increase performance
of the model for future optimizations.  The solver for the algorithm was kept at
the default 'liblinear' function which is most useful against smaller datasets,
such as the one in this project.  

Results were visualized using graphs generated with the
[matplotlib](http://matplotlib.org/index.html)   package.  Performance of the
logistic regression model was tested against the held-out test set using the
[log
loss](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss)
metric.  For reference, the [F2
score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html)
and [Matthews Correlation
Coefficient](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html)
scores are also listed, though they describe the performance of the algorithm to
correctly classify metastasis state.  Both the graphical analysis and metric
reports were generated for each testing cycle using the scripts supplied in the
['Support
Files'](https://github.com/CCThompson82/MLE_capstone/tree/master/Support%20Files)
folder in the GitHub repository.  

## Refinement

In order to optimize the C parameter, or to determine whether the default was
indeed optimal,  a [Logistic
Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV)
classifier generated using 4 fold cross-validation across a 10-log range for C.
Performance was measured using [log
loss](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss)
as the  scoring function.

The next step in model optimization was to feed back the Gleason score, shown to
be the most important explanatory variable in the benchmark analysis.  It was
unclear whether adding another feature would contribute to variance or improve
generalization. As the coefficients for the 2nd and 3rd principle component were
significantly less than the coefficients for the first principle component and
Gleason score, they were removed from the final model to achieve an incremental
increase in model performance.  An alternative to this approach would have been
to change the regularization function to 'l1', which would have the effect to
silence the contribution of the less-important features on calculation of the
dependent variable.  

# Results

## Model Evaluation and Validation

### Final Model

The final logistic regression model receives 2 feature variables:

1. Gleason score
2. the first principle component from a PCA transformed subset of 20 gene activation values

The coefficients for these features were routinely equivalent, indicating they
contribute roughly evenly to dependent variable prediction.  The optimal
regularization parameter was regularly determined as the maximum value tested,
which is an indication of noisy (_i.e._ not linearly separable) data.

![Figure 7](/Figures/final_figure.png)

**Figure 7 ** - Summary of the change in metric score over the optimization course of the project.  The final model, that incorporates a single principle component with Gleason score performs better than the benchmark model in three metrics tested.

The error / accuracy rate of three performance metrics was often similar
between the training sample and test sample set predictions, indicating the
model was not over-fit.  As only 2 feature  variables were incorporated into the
training of the final model, the possibility of bias was present.  However

### Test set Validation

**Table 1**- Performance across 5 random seeds  

| Seed    | LogLoss   | Benchmark_LogLoss   | %_Improvement_over_benchmark |
|---------|-----------|---------------------|-----------------------------|
| 1       | 0.508049  | 0.594456            | 14.5 |
|12      |0.477352  |0.570673            |16.5
|123     |0.474036  |0.644465            |26.3
|1234    |0.467069  |0.617577            |24.3
|12345   |0.468992  |0.606186            |22.6


This project's strategy was to leave out 30% of the original dataset to use as a
true validation of the models' generalization capability.  The final model test
set log loss score ranged from 0.467 to 0.508 across five different random state
seeds, indicating that modification of the training and test set samples did not
affect outcome of model performance.

## Justification

The final logistic regression model was always more accurate in predicting the
probability of prostate cancer  metastasis than the benchmark model.  Over the
five consecutive runs described above, an average improvement of 20.8% in log
loss score was achieved over the starting benchmark score.  


# Conclusion

## Free-Form Visualization

A function was written that accepts an RNA-seq gene count profile (in TPM
format), and outputs the model risk for metastasis.  This function was applied
to every sample for which no label was given and showed that a significant
portion of patients in the cohort exhibit a high level of risk for metastasis.   

![Figure 8](/Figures/Label_missing.png)

**Figure 8** - Metastasis predictions for unlabeled TCGA cohort samples.
TCGA cohort patient samples that did not include a metastasis
label and were Gleason range 7-10 were omitted from model learning and validation.
Samples are subjected to the risk analysis function and plotted against the
benchmark model prediction (left) and Gleason score (right).


![Figure 9](/Figures/n0_re-analysis.png)

**Figure 9** - Distribution of metastis probability for samples labeled and
presumed to be non-metastatic.  Many examples are predicted to have a high likelihood
of metastasis.  


## Reflection

### Objective

The purpose of this project was to generate a model capable of supplying a
patient and doctor with a metric for risk of Prostate Cancer metastasis that was
more useful than simple use of the 'Gleason Score'.  To accomplish this, RNA-seq
(gene activation profile) was explored as a potential inroad into personalized
therapy for newly diagnosed Prostate cancer patients.  There were several issues
that made this task difficult:
1. Small, wide sample data - the effective
dataset (containing Gene Activation profile and a metastasis label) was 446
samples by 20501 gene features.   
2. 'Inaccurate' / 'Pre-mature' data labelling -
The TCGA cohort is regularly updated and those listed as non-metastatic at the
time of update could become metastatic at a later date.
3. Noise in the data -
no single gene or biomarker had been reported as capable of efficiently
separating non-metastatic and metastatic cancers(corroborated in this project,
**Figure 10**).


Thus from a machine learning perspective, it was clear from the project's onset
that feature reduction and appropriate model selection would be paramount to
success.  

### Feature Selection

There are many techniques for feature reduction.  One avenue explored was
feature elimination via a wrapping mechanism.  However this approach was very
slow and provided inconsistent results in which features and how many features,
were important. A different approach, which was successful, was to utilize the
training of an ensemble Random Forest classifier, not for its use in
classification, but in order to access its assessment of which genes were most
informative in separation of the metastasis classes.  Despite the inherent
random sampling employed by this approach, results were largely stable with
10-15 genes returned in the top ranked 20 in almost every run, across several
random seeds.  

Visualized individually, none of these 20 genes could separate the metastasis
state linearly.

![Figure 10](/Figures/Gene_separation.png)

**Figure 10** - Genes with the highest 'Gini Impurity' score were not able to separate
metastasis class linearly.  

However, when compressed into principle components, this 20 feature set became
predictive.  I chose to retain the top 3 principle components of the 20-feature
set, due to the noise levels expected from the small dataset (at 3 features,
this left ~110 examples per feature in the training dataset).  Curiously, with
this approach the first principle component (labeled '0' in the notebook and
relevant figures) always

The initial plan was to provide the full  complement of principle components
(originally, 20) to the logistic regression classifier as training data, and
subsequently use each component's coefficient to  assess which were most able to
explain the independent variable in a [Recursive
Elimination]('http://scikit-learn.org/0.15/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV')
wrapping function.  However, graphical analysis of the PCA transformed dataset
revealed that the first principle component clearly separated the two metastasis
states into nearly distinct Gaussian distributions, despite the fact that PCA is
an unsupervised learning algorithm.  The same result  was observed irrespective
of whether 10, 20, 50, 100, 200, or 400 genes were retained from the Gini
Impurity filter step.  

How could this be?  This result would be expected if a transformation technique
such as linear discriminant analysis (LDA) had been employed, as LDA uses data
label in order to determine the component vectors where class label is
discriminated the most.  PCA, on the other hand, is an unsupervised technique
and had generated what appeared to be a discriminant component in the absence of
label information.  However, upon reflection, it is perhaps not surprising that
the eigenvector where the most variance in the data subset was contained (_i.e._
the first principle component) would separate the class labels, given that
_only genes where a 'significant' difference in gene expression between the
class labels_ were retained and provided to the PCA model.  

By creating a pipeline from the  Gini Importance filter directly into the PCA
transformation, something similar to [Linear Discriminant
Analysis](http://scikit-learn.org/0.16/modules/generated/sklearn.lda.LDA.html)
had been generated.  Indeed, exploration of an supervised LDA compression of the
20-feature set yielded a similar level of performance in the final model
compared to compression via Gini Impurity to PCA pipeline.

![Figure 11](/Figures/PC_components_scatter_matrix.png)

**Figure 11** - Analysis of PCA transformation of a 20-gene feature subset.  The
first principle component of PCA transformation separates metastasis state more
efficiently than any single gene from the input set.  The second and third principle components are also shown for reference.  

The 3-component feature set was split on the same indices that were generated in
the training and validation sets used in the benchmark analysis.  This was done
to aid in model to model comparisons within each run.  To note, this split was
originally stratified on the y-label (metastasis state).  However, after
observing moderately inconsistent results for final model validation
performance, the decision was made to stratify by Gleason score (Cancer
severity) of the samples.   This decision ensured that difficult cases - those
in the middle range of severity - were equivalently distributed among the
training and test sets, vastly reducing the run to run variation.  

### Model Selection

 Having completed a feature selection and compression technique, in which at
 least the first principle component seemed capable of distinguishing among
 metastasis class via graphical analysis (**Figure 11**), a logistic regression
 classifier was chosen as the predictive model.  Logistic regression was
 preferred  to other hyperplane-based techniques, such as support vector
 machines (SVM) due to the noise that was expected in the compressed dataset.
 SVM classifiers attempt to define the hyperplane by which the margin between
 the class labels is maximized.  In situations where data is not easily
 separable, this result can be unstable, and  at times, arbitrary.  Moreover,
 SVM does not provide a true probability of class assignment, as was the
 objective of the project.  In contrast, logistic regression assumes that no
 feature  is capable of explaining the outcome variable, but that the
 combination of features  should be able to provide a probability of class
 assignment.  This assumption holds true for the RNA-seq dataset employed in
 this project.  Moreover, as the objective of this project was to provide a
 probability of metastasis, the output of logistic regression classifier is
 perfectly suited.  

### Training and Optimization

Separate Train and Test indices were stratified based on cancer severity prior
to the benchmark analysis and the final PCA compressed (3-components) were
subset into these indices.  Logistic regression classifier was trained and
optimized on the Train set, prior to validation on the Test set.  Gleason score
was added as a feature to this model and the 2nd and 3rd principle components
were eliminated from the model after determining that they did not contribute to
model performance.  

The final release version of the code was run across 5 seeds and performance in
the primary metric (log loss), and secondary metrics (F2 and MCC) were recorded,
compared to the benchmark.

### Model Performance

In every run tested, the performance of the final model exceeded performance of
the benchmark model by at least 14% in log loss score.  The pipeline exhibited
in this project could be re-appropriated for other types of RNA-seq based
classifications.  By looking for individual genes whose activation level explain
a certain condition, researchers may be missing the opportunity to provide
valuable disease prognosis.  Instead, by performing a feature selection and
compression, researchers may be able to predict disease more regularly at the
sacrifice of knowing _exactly_ what genes are causal.  

Importantly, I hypothesize that as the TCGA cohort study is updated
longitudinally, its performance will be more accurate.  This is due to the
nature  of analyzing an on-going cohort trial.  In the context of a machine
learning problem,  sample labels will only move in one direction (from
'non-metastatic' to 'metastatic', never _vis-a-versa_).  Therefore those patient
samples predicted with a high  probability of metastasis, currently labeled as
non-metastatic, would be correctly classified in future validation.  

Unfortunately in the context of the TCGA cohort study, the link between patient
and barcode has been broken for ethical reasons, meaning that such patients with
high risk  can not be identified for extra care in monitoring metastasis.  


## Improvement

There is still bias in this model due to the small sample size.  Increased
number of specimens could allow more resolution / stability in feature selection
and  compression.  For each iteration of the code, a handful of genes selected
from the 'Random Forest filter' is altered, though 10-15 remain identical.  

Evidence here and elsewhere suggests that no gene or principle component could
be capable of separating  metastasis state classes, and thus logistic regression
is an excellent long term  model for prediction.  However, it is possible that
other sources of information  could help improve model accuracy, including
genetic or epigenetic specimen data.  Ultimately a great many more number of
cases will be required to increase the  resolution for metastasis prediction.  










[End]
