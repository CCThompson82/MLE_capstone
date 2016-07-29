fig = plt.figure(figsize=(15,10))
A= fig.add_subplot(2,2,1)
B = fig.add_subplot(2,2,2)
A.scatter(bench_risk,
          model_risk,
          color = 'green',
          alpha = 1,
          s = 25)
B.scatter(clinicalDF_all.loc[not_labeled.index, 'gleason'],
          model_risk,
          color = 'green',
          alpha = 1,
          s = 25)
for x in [A,B] :
    x.set_ylim(0,1)
    x.set_ylabel('Probability of Metastasis')
fig.suptitle('Metastasis predictions for unlabeled TCGA cohort samples', fontsize=18)
A.set_xlabel('Benchmark Model probability')
B.set_xlabel('Gleason Score')
fig.savefig('Figures/Label_missing.png')
plt.show
