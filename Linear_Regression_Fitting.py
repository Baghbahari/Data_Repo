import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

q1x = np.array(pd.read_csv('q2x.txt'))
Q1xx = []
Q1xy = []
for i in range(0, (q1x.shape)[0]):
	Elmn = q1x[i][0]	
	elmn = float(Elmn)
	Q1xx.append(elmn)

q1y = np.array(pd.read_csv('q2y.txt'))
Q1y = []
X = []
for i in range(0, (q1y.shape)[0]):
	Elmn = q1y[i]#[0][0:13]	
	elmn = float(Elmn)
	Q1y.append(elmn)
	X.append([1, Q1xx[i]])
#print(Q1xx, Q1y)

### Regression
X= np.array(X)
Psedu = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T) 
w = np.dot(Psedu, np.array(Q1y))
print(np.linalg.norm(np.array(Q1y) - np.dot(X, np.array([+0.32767539, 0.17531122]) * len(Q1y)),2))
print(w)
plt.scatter(Q1xx,Q1y)
#plt.show()
x1 = np.linspace(min(Q1xx), max(Q1xx))
#print(x1)
x2 = +(w[1])*x1 + (w[0])
#print(x1,x2)

plt.plot(x1,x2,'r')
plt.show()
