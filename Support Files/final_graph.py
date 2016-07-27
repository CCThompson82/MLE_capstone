fig = plt.figure(figsize=(8,8))
A = fig.add_subplot(1,1,1)
ya = 0.2
yb = 0.8
A.set_ylim(ya,yb)
A.set_xlim(-0.5,4.5)
A.xaxis.set_major_formatter(plt.NullFormatter())
for i,x in enumerate(metric_stat.index) :
    A.text(i-.25,ya-((yb-ya)/20),x, rotation = -45, fontsize=14)
#A.text([0,1,2,3], [1,1,1,1], [metric_stat.index])
A.plot(range(0,metric_stat.shape[0],1),
          metric_stat['F2'],
          color = 'blue', label = 'F2 score',
          marker= 'o')
A.plot(range(0,metric_stat.shape[0],1),
          metric_stat['MCC'],
          color = 'green', label = 'MCC score',
          marker= 'o')
A.plot(range(0,metric_stat.shape[0],1),
          metric_stat['LogLoss'],
          color = 'black', label = 'LogLoss score',
          marker= 'o')
A.legend(loc=2)
