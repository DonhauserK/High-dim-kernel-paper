"""
This script runs CCM on three example datasets.
These datasets correspond to the synthetic datasets in the paper.

Type of the datasets are binary classification,
categorical classification, and regression respectively.
"""
from __future__ import print_function
import numpy as np
import sys
sys.path.append('core')


# We use https://github.com/Jianbo-Lab/CCM
import ccm
import math



import scipy.io
# https://jundongl.github.io/scikit-feature/datasets.html
mat = scipy.io.loadmat('CLL_SUB_111.mat')

X = np.array(mat["X"])
Y = np.reshape(np.array(mat["Y"]),(-1))
ydiscard = Y != 1
X = X[ydiscard,:]
Y = Y[ydiscard]
Y = Y -2
print(Y.shape)




nfeatures = np.shape(X)[1]
epsilon = 0.0001; num_features = 100; type_Y = 'binary'

features_idcs = []
for i in range(math.ceil(nfeatures/1000)):
	Xt = X[:,i*(1000):-1]
	Xt2 = Xt[:,0:min(np.shape(Xt)[1],1000)]
	num_features = min(np.shape(Xt)[1],100)
	rank = ccm.ccm(Xt2, Y, num_features, type_Y,
		epsilon, iterations = 1000)
	features_idcs.append(i*500+ np.argsort(rank)[:min(np.shape(Xt)[1],100)])
	print("done with round ")
	print(str(i))

print(np.concatenate(features_idcs))
features_idcs_ls= np.concatenate(features_idcs)
X2 = X[:,features_idcs_ls]
nfeatures = np.shape(X2)[1]
print(nfeatures)
print("number of features ")
epsilon = 0.0001; num_features = 100; type_Y = 'binary'

features_idcs2 = []
for i in range(math.ceil(nfeatures/1000)):
	Xt = X2[:,i*(1000):-1]
	Xt2 = Xt[:,0:min(np.shape(Xt)[1],1000)]
	num_features = min(np.shape(Xt)[1],100)

	rank = ccm.ccm(Xt2, Y, num_features, type_Y,
		epsilon, iterations = 1000)
	features_idcs2.append(i*500 + np.argsort(rank)[:min(np.shape(Xt)[1],100)])
	print("done with round 2")
	print(str(i))
features_idcs_ls2= np.concatenate(features_idcs2)

X3 = X2[:,features_idcs_ls2]
nfeatures = np.shape(X2)[1]
epsilon = 0.0001; num_features = 200; type_Y = 'binary'
num_features = min(np.shape(X3)[1],100)

rank = ccm.ccm(X3, Y, num_features, type_Y,
	epsilon, iterations = 1000)
features_idcs2.append(np.argsort(rank))
print("done with round 3 ")
features_idcs3 = np.argsort(rank)[:min(np.shape(X2)[1],200)]

np.save("features1_cll_sub.npy",np.array(features_idcs_ls))
np.save("features2_cll_sub.npy",np.array(features_idcs_ls2))
np.save("features3_cll_sub.npy",np.array(features_idcs3))
