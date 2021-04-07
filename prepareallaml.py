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
#https://jundongl.github.io/scikit-feature/datasets.html
mat = scipy.io.loadmat('ALLAML.mat')

X = np.array(mat["X"])
Y = np.reshape(np.array(mat["Y"]),(-1))
# ydiscard = Y != 1
# X = X[ydiscard,:]
# Y = Y[ydiscard]
# Y = Y -2
print(Y.shape)


nf = 200
ntotal = 2000

nfeatures = np.shape(X)[1]
epsilon = 0.1; num_features = nf; type_Y = 'binary'

features_idcs = []
for i in range(math.ceil(nfeatures/ntotal)):
	Xt = X[:,i*(ntotal):-1]
	Xt2 = Xt[:,0:min(np.shape(Xt)[1],ntotal)]
	num_features = min(np.shape(Xt)[1],nf)
	rank = ccm.ccm(Xt2, Y, num_features, type_Y,
		epsilon, iterations = ntotal)
	features_idcs.append(i*500+ np.argsort(rank)[:min(np.shape(Xt)[1],nf)])
	print("done with round ")
	print(str(i))

print(np.concatenate(features_idcs))
features_idcs_ls= np.concatenate(features_idcs)
X2 = X[:,features_idcs_ls]
nfeatures = np.shape(X2)[1]
print(nfeatures)
print("number of features ")
epsilon = 0.0001; num_features = nf; type_Y = 'binary'

features_idcs2 = []
for i in range(math.ceil(nfeatures/ntotal)):
	Xt = X2[:,i*(ntotal):-1]
	Xt2 = Xt[:,0:min(np.shape(Xt)[1],ntotal)]
	num_features = min(np.shape(Xt)[1],nf)

	rank = ccm.ccm(Xt2, Y, num_features, type_Y,
		epsilon, iterations = ntotal)
	features_idcs2.append(i*500 + np.argsort(rank)[:min(np.shape(Xt)[1],nf)])
	print("done with round 2")
	print(str(i))
features_idcs_ls2= np.concatenate(features_idcs2)

# X3 = X2[:,features_idcs_ls2]
# nfeatures = np.shape(X2)[1]
# epsilon = 0.0001; num_features = 200; type_Y = 'binary'
# num_features = min(np.shape(X3)[1],nf)
#
# rank = ccm.ccm(X3, Y, num_features, type_Y,
# 	epsilon, iterations = ntotal)
# features_idcs2.append(np.argsort(rank))
# print("done with round 3 ")
# features_idcs3 = np.argsort(rank)[:min(np.shape(X2)[1],200)]

np.save("allaml1.npy",np.array(features_idcs_ls))
np.save("allam2.npy",np.array(features_idcs_ls2))
