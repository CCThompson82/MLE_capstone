"Performance on Test dataset"
print('Model:\n',clf_LR)
print('\n',classification_report(y_test, clf_LR.predict(X_test), labels = ['n0','n1']))
metric_stat = metric_stat.append(pd.DataFrame({'F2': fbeta_score(y_test, clf_LR.predict(X_test), pos_label='n1',beta=2),
                                 'MCC': matthews_corrcoef(y_test, clf_LR.predict(X_test)),
                                 'LogLoss':log_loss(y_test.replace({'n0':0, 'n1':1}),
                                           clf_LR.predict_proba(X_test)[:,1])}, index = [run_name]),
                                           ignore_index=False)
print('\nLog Loss: ',log_loss(y_test.replace({'n0':0, 'n1':1}),
          clf_LR.predict_proba(X_test)[:,1]))
print("\n\n",metric_stat)

training_stat = training_stat.append(pd.DataFrame({'F2': fbeta_score(y_train, clf_LR.predict(X_train), pos_label='n1',beta=2),
                                 'MCC': matthews_corrcoef(y_train, clf_LR.predict(X_train)),
                                 'LogLoss':log_loss(y_train.replace({'n0':0, 'n1':1}),
                                           clf_LR.predict_proba(X_train)[:,1])}, index = [run_name]),
                                           ignore_index=False)
