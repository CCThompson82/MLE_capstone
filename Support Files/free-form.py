fig = plt.figure(figsize=(15,10))
A= fig.add_subplot(2,2,1)
B = fig.add_subplot(2,2,2)
A.scatter(bench_risk,
          model_risk,
          color = 'black',
          alpha = 1,
          s = 25)
B.scatter(clinicalDF_all.loc[not_labeled.index, 'gleason'],
          model_risk,
          color = 'black',
          alpha = 1,
          s = 25)
for x in [A,B] :
    x.set_ylim(0,1)
    x.set_ylabel('Probability of Metastasis')
fig.suptitle('Metastasis predictions for unlabeled TCGA cohort samples', fontsize=24)
A.set_xlabel('Benchmark Model probability')
B.set_xlabel('Gleason Score')
plt.show()
plt.savefig('Figures/Label_missing.png')


fig = plt.figure(figsize=(10,5))
A = fig.add_subplot(1,1,1)
A.hist(n0_risk.sort_values(ascending=False),
       bins=21,
       facecolor='blue')
#A.set_yscale('log')
A.set_ylabel('Count', fontsize=16)
A.set_xlabel("Model Probability of Metastasis", fontsize=16)
fig.suptitle("Distribution of metastasis probability for samples presumed to be local malignancies", fontsize = 18)
plt.show()
plt.savefig('Figures/n0_re-analysis.png')
