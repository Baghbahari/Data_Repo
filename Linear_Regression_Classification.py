import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

q1x = np.array(pd.read_csv('q1x.txt'))
Q1xx = []
Q1xy = []
for i in range(0, (q1x.shape)[0]):
	Elmn = q1x[i][0][0:13]	
	elmn = float(Elmn)
	Q1xx.append(elmn)
	Elmn = q1x[i][0][15:]
	elmn = float(Elmn)
	Q1xy.append(elmn)

q1y = np.array(pd.read_csv('q1y.txt'))
Q1y = []
X = []
for i in range(0, (q1y.shape)[0]):
	Elmn = q1y[i]#[0][0:13]	
	elmn = 2*float(Elmn) - 1
	Q1y.append(elmn)
	X.append([1, Q1xx[i], Q1xy[i]])
X= np.array(X)
Psedu = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T) 
w = np.dot(Psedu, np.array(Q1y))

print(w)
x1 = np.linspace(min(Q1xx), max(Q1xx))
#print(x1)
x2 = -(w[2]/w[1])*x1 - (w[0]/w[1])
#print(x1,x2)
plt.scatter(Q1xx[0:49],Q1xy[0:49])
plt.scatter(Q1xx[49:],Q1xy[49:])
plt.plot(x1,x2)
plt.show()
