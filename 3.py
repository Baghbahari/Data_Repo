#!/usr/bin/env python                                                                                     
#-*- coding:utf-8 -*- 
import pandas as pd
import numpy as np
from svmutil import *
import matplotlib.pyplot as plt
from sklearn import svm
import csv
import random
from sklearn.cross_validation import KFold
import pickle
from tempfile import TemporaryFile

def Nrmlz(data):
    u = np.mean(data)#float(sum(data))/float(len(data))
    data_u = np.array(data) - u
    Sig = np.var(data_u)
    data_n = (data - np.min(data)) / (np.max(data) - np.min(data))#data_u/Sig
    return data_n


yeast_data = np.array(pd.read_csv('yeast.txt'))
feat = []
label = []
F1 = []
F2 = []
F3 = []
F4 = []
F5 = []
F6 = []
F7 = []
F8 = []
for i in range(0, (yeast_data.shape)[0]):
	data = yeast_data[i][0].split()
	F = [float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5]), float(data[6]), float(data[7]), float(data[8])]
	if data[-1] == 'CYT':
		label.append(1)
	else:
		label.append(-1)	
	F1.append(F[0])
	F2.append(F[1])
	F3.append(F[2])
	F4.append(F[3])
	F5.append(F[4])
	F6.append(F[5])
	F7.append(F[6])
	F8.append(F[7])
	feat.append(F)
F = [F1,F2,F3,F4,F5,F6,F7,F8]

D = []
#Feat = F1.append(F2)
for i in range(0,len(F)):
	Fi = Nrmlz(F[i])	
	D.append(Fi.tolist())

	
print(Nrmlz(F[1]))
D = np.array(D).T.tolist()#np.array(F).T.tolist()#

plt.plot(Nrmlz(F[1]))
#plt.show()

kf = int(round(len(label)/1,-1))
RND = []
Rnd = len(label)
Nf = 10
arr = np.arange(Nf)
#idcs = np.random.permutation(kf) # 	rnd = np.random.choice(Rnd, 100,  replace=False)
IDC = []
'''
f = open('store.pckl', 'wb')
pickle.dump(idcs, f)
f.close()
'''
f = open('store.pckl', 'rb')
idcs = pickle.load(f)
f.close()

for k in range(0,Nf):
	Idcs = idcs[k*kf/Nf:(k+1)*kf/Nf]
	IDC.append(Idcs)

svm_model.predict = lambda self, x: svm_predict([0], [x], self)[0][0]




Dtst = []
Dtrn = []
Ltst = []
Ltrn = []

EE = []
Nc = 12
#for d in range(1,5):
d = 3
Erm = []
Esig = []
#for c in range(-1,Nc):
#for d in range(1,5):
Err = []
for i in range(0,len(IDC)):			
	Dtst = []
	Dtrn = []
	Ltst = []
	Ltrn = []
	#Err = []
	for k,idx in enumerate(IDC[i]):
		Dtst.append(D[idx])
		Ltst.append(label[idx])
	Trn_idx = np.delete(idcs,IDC[i])
	for k,idx in enumerate(Trn_idx):
		Dtrn.append(D[idx])
		Ltrn.append(label[idx])
	problem = svm_problem(Ltrn, Dtrn)
	parameter = svm_parameter('-t 1 -d 1 -c 1024')
	m = svm_train(problem, parameter)
	'''
	prob = svm_problem(Ltrn, Dtrn)
	param = svm_parameter()
	c = 10
	param.C = 2**c
	#param.h = 0
	param.t = 2
	#param.g = 1
	#param.v = 0
	param.g = 5
	m = svm_train(prob, param)
	'''
	result_tst = svm_predict(Ltst, Dtst , m)
 	diff_tst = np.sum(np.abs(np.array(result_tst[0]) - np.array(Ltst)))/2
	err_tst = diff_tst/len(Ltst)
	result_trn = svm_predict(Ltrn, Dtrn , m)
 	diff_trn = np.sum(np.abs(np.array(result_trn[0]) - np.array(Ltrn)))/2
	err_trn = diff_trn/len(Ltrn)
	Err.append([err_trn, err_tst])
print(Err)
'''
Esig.append(np.var(Err))
Erm.append(np.mean(Err))
EE.append([Erm,Esig])
print(EE)
f = 0
#for f in range(len(EE)):
fig = plt.figure()
plt.plot([x for x in range(-1,Nc)],EE[f][0])
plt.plot([x for x in range(-1,Nc)],np.array(EE[f][0])-np.array(EE[f][1]))
plt.plot([x for x in range(-1,Nc)],np.array(EE[f][0])+np.array(EE[f][1]))
plt.show()
'''

'''
with open("DATA", "w") as output:
	writer = csv.writer(output, delimiter='\t')
	writer.writerows(zip(F1,F2))
'''

#plt.plot((F5))
#plt.show()
'''
ER = []
for c in range(-2,20):
	clf = svm.SVC(C=2**c, kernel='poly', degree=3)
	#clf.kernel = 'polynomial'
	#clf.degree = 1
	L = 1300
	clf.fit(D[:L], label[:L])
	#print(clf)
	tr = label[L:]
	error = clf.predict(D[L:])
	diff = np.sum(np.abs(np.array(error) - np.array(tr)))/2
	ER.append(float(diff)/len(tr))
	print(float(diff)/len(tr))
plt.plot(ER)
plt.show()
'''
'''
prob = svm_problem(label[:1335], D[:1335])
#param.n = 10
#param.shrinking = 0
err = []
for c in range(-2,15):
	print(c)
	param.C = 2**c
	param.h = 0
	param.t = 1
	param.g = 1
	param.v = 0
	param.d = 1
	m = svm_train(prob, param)

	p_label = [1, 1, -1, -1]
	p_data = [
	    [0.3, 0.9, 1.2],
	    [2.0, 3.0, 4.5],
	    [3.0, 1.0, 0.3],
	    [1.0, 0.5, 0.25]
	    ]
	result = svm_predict(label[1335:], D[1335:] , m)
	#result_string = result.getvalue()
 	diff = np.sum(np.abs(np.array(result[0]) - np.array(label[1335:])))/2
	err.append(diff)
	print(diff/len(label[1335:]))
	#cmd = [ 'svm-train', '-c', '1', '-t', '1' , 'DATA.scale']
	#output = subprocess.Popen( cmd, stdout=subprocess.PIPE ).communicate()[0]

plt.plot([i for i in range(-2,15)], err)
plt.show()
'''

'''
from svm import *
from svmutil import *

# For learning                                                                                            
t_label = [1,-1,1,-1]
t_data = [
    [1.0, 2.0, 3.0],
    [3.0, 1.5, 1.0],
    [2.0, 3.0, 4.0],
    [0.5, 1.0, 1.5]
    ]
problem = svm_problem(t_label, t_data)
parameter = svm_parameter('-s 0 -t 0')
t = svm_train(problem, parameter)

# For predict                                                                                             
p_label = [1, 1, -1, -1]
p_data = [
    [0.3, 0.9, 1.2],
    [2.0, 3.0, 4.5],
    [3.0, 1.0, 0.3],
    [1.0, 0.5, 0.25]
    ]
result = svm_predict(p_label, p_data , t)

print "[Result]"
for r in result:
    print r
'''
