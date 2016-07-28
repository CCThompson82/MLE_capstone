fig = plt.figure(figsize=(12,6))
A = fig.add_subplot(1,2,1)
B = fig.add_subplot(1,2,2)
ya = 0.2
yb = 0.9
A.set_ylim(ya,yb)
A.set_xlim(-0.5,4.5)
A.xaxis.set_major_formatter(plt.NullFormatter())
for i,x in enumerate(metric_stat.index) :
    A.text(i-.25,ya-((yb-ya)/20),x, rotation = -45, fontsize=14)

B.set_ylim(ya,yb)
B.set_xlim(-0.5,4.5)
B.xaxis.set_major_formatter(plt.NullFormatter())
for i,x in enumerate(metric_stat.index) :
    B.text(i-.25,ya-((yb-ya)/20),x, rotation = -45, fontsize=14)


A.plot(range(0,metric_stat.shape[0],1),
          metric_stat['F2'],
          color = 'blue', label = 'F2 test score',
          marker= 'o')
A.plot(range(0,training_stat.shape[0],1),
          training_stat['F2'],
          color = 'blue',
          label = 'F2 training score',
          linestyle ='--')
A.plot(range(0,metric_stat.shape[0],1),
          metric_stat['MCC'],
          color = 'green', label = 'MCC test score',
          marker= 'o')
A.plot(range(0,training_stat.shape[0],1),
          training_stat['MCC'],
          color = 'green',
          label = 'MCC training score',
          linestyle ='--')
B.plot(range(0,metric_stat.shape[0],1),
          metric_stat['LogLoss'],
          color = 'black', label = 'LogLoss test score',
          marker= 'o')
B.plot(range(0,training_stat.shape[0],1),
          training_stat['LogLoss'],
          color = 'black',
          label = 'LogLoss training score',
          linestyle ='--')
A.legend(loc=2)
B.legend(loc=2)
A.set_ylabel('Metric Score', fontsize=16)
fig.suptitle('Summary of Model Performance', fontsize = 20)
plt.show()
