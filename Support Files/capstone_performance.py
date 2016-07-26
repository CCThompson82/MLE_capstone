"Performance on Test dataset"
print('Model:\n',clf_LR)
print('\nF2 score: ', fbeta_score(y_test, clf_LR.predict(X_test), pos_label='n1',beta=2))
print('\nMCC: ',matthews_corrcoef(y_test, clf_LR.predict(X_test)),"\n")
print(classification_report(y_test, clf_LR.predict(X_test), labels = ['n0','n1']))
print('\nLogLoss: ', log_loss(y_test.replace({'n0':0, 'n1':1}),
         clf_LR.predict_proba(X_test)[:,1]))
