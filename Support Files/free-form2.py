

fig = plt.figure(figsize=(10,5))
A = fig.add_subplot(1,1,1)
A.hist(n0_risk.sort_values(ascending=False),
       bins=21,
       facecolor='blue')
#A.set_yscale('log')
A.set_ylabel('Count', fontsize=16)
A.set_xlabel("Model Probability of Metastasis", fontsize=16)
fig.suptitle("Distribution of metastasis probability for samples presumed \n to be local malignancies", fontsize = 14)
plt.show()
