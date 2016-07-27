fig = plt.figure(figsize=(15,8))
A= fig.add_subplot(2,2,1)
B = fig.add_subplot(2,2,2)
A.scatter(clf_LR_clin.predict_proba(clinicalDF_train)[:,1],
          clf_LR.predict_proba(X_train)[:,1],
          color = y_train.replace({'n1':'red', 'n0': 'blue', 'NaN':'grey'}),
          alpha = 0.4,
          s = 25)
A.set_ylabel('Model probability of metastasis')
A.set_xlabel('Benchmark model probability of metastasis')
#plt.suptitle('Prognosis of metastasis risk based on features available at presentation', fontsize=20)

B.scatter(clf_LR_clin.predict_proba(clinicalDF_test)[:,1],
          clf_LR.predict_proba(X_test)[:,1],
          color = y_test.replace({'n1':'red', 'n0': 'blue', 'NaN':'grey'}),
          alpha = 0.5,
          s = 25)
A.set_ylim(0,1.05)
A.set_xlim(0,1)
A.set_title('Training Set')
B.set_title('Test Set')
B.set_ylabel('Model probability of metastasis')
B.set_xlabel('Benchmark model probability of metastasis')
B.set_ylim(0,1.05)
B.set_xlim(0,1)

C = fig.add_subplot(2,2,3)
D = fig.add_subplot(2,2,4)

C.scatter(clinicalDF_train.loc[:,'gleason'],
          clf_LR.predict_proba(X_train)[:,1],
          color = y_train.replace({'n1':'red', 'n0': 'blue', 'NaN':'grey'}),
          alpha = 0.4,
          s = 25)
C.set_xlabel('Gleason score')
C.set_ylabel('Model probability of metastasis')
#plt.suptitle('Prognosis of metastasis risk based on features available at presentation', fontsize=20)

D.scatter(clinicalDF_test.loc[:,'gleason'],
          clf_LR.predict_proba(X_test)[:,1],
          color = y_test.replace({'n1':'red', 'n0': 'blue', 'NaN':'grey'}),
          alpha = 0.5,
          s = 25)
C.set_ylim(0,1.05)
C.set_xlim(5,11)
C.set_ylabel('Model probability of metastasis')
D.set_xlabel('Gleason score')
D.set_ylim(0,1.05)
D.set_xlim(5,11)
fig.legend(labels = ['n0', 'n1'], handles= [blue,red], loc = 5)
#B.legend()
#bench_fig.title("Prediction of Metastasis")
#plt.show
#bench_fig.savefig('Figures/gleason_F3.png')
plt.show
#fig.savefig('Figures/bench_F3.png')
