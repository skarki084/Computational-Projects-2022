# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 22:56:32 2024

@author: samip
"""

import numpy as np
import struct
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy as sp


with open('data/train-images.idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    Xtraindata = np.transpose(data.reshape((size, nrows*ncols)))

with open('data/train-labels.idx1-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    ytrainlabels = data.reshape((size,)) # (Optional)

with open('data/t10k-images.idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    Xtestdata = np.transpose(data.reshape((size, nrows*ncols)))

with open('data/t10k-labels.idx1-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    ytestlabels = data.reshape((size,)) # (Optional)
        

    
traindata_imgs =  np.transpose(Xtraindata).reshape((60000,28,28))    
#print(Xtraindata.shape)
#print(ytrainlabels.shape)
#print(Xtestdata.shape)
#print(ytestlabels.shape)


def plot_digits(XX, N, title):
    fig, ax = plt.subplots(N, N, figsize=(8, 8))
    
    for i in range(N):
      for j in range(N):
        ax[i,j].imshow(XX[:,(N)*i+j].reshape((28, 28)), cmap="Greys")
        ax[i,j].axis("off")
    fig.suptitle(title, fontsize=24)
    plt.show()

plot_digits(Xtraindata, 8, "First 64 Training Images" )



# #####################################
#Task 1: I want to take the columns of Xtrain and do PCA analysis

Xtraindata_mean_0 =Xtraindata - np.mean(Xtraindata,axis = 1, keepdims=True)
dU, ds, dVt = sp.linalg.svd(Xtraindata_mean_0, full_matrices=False)


def plot_PCs(XX, N, title):
    fig, ax = plt.subplots(N, N, figsize=(8, 8))
    
    for i in range(N):
      for j in range(N):
        ax[i,j].imshow(XX[:,(N)*i+j].reshape((28, 28)), cmap="Greys")
        ax[i,j].axis("off")
    fig.suptitle(title, fontsize=24)
    plt.show()

plot_PCs(dU[:,0:16], 4, 'First 16 PC Modes')


#Task 2: inspect cumulative energy of the singular values, and determine
#the number of PC modes needed to approximate 85% of the energy
E = np.power(ds,2)/np.sum(np.power(ds,2))
cumsum = np.cumsum(E)
plt.figure(figsize=(8, 8))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.plot(cumsum[0:100])

plt.title('Cumulative Sum of Energy of Principal Components', fontsize = 24)
plt.xlabel('index $j$', fontsize = 24)
plt.ylabel('$\Sigma E_j$' ,fontsize = 24)
plt.show()

#59 PC modes are needed to get 85% energy

k = 59
dU_trunc = dU[:,0:k] #only k PC modes

#Task 3:  Write a function that selects a subset of particular digits
#and then returns the subset as new matrices
#So I call the function, input: digit, then return Xsubtrain, Ysubtrain, Xsubtest, Ysubtest

def subset(n):
    train_index = np.where(ytrainlabels ==n)[0]
    test_index = np.where(ytestlabels == n)[0]
    
    Xsubtrain = Xtraindata[:,train_index]
    ysubtrain = ytrainlabels[train_index]
    Xsubtest = Xtestdata[:, test_index]
    ysubtest = ytestlabels[test_index]
    
    return [Xsubtrain, ysubtrain,Xsubtest,ysubtest]
    
#Task 4: use this function to get all the 1s and 8s from all the data
#Project the X data into k-PC modes (k= 59)
#use ridge classification, see testing accuracy and cross validation
[Xsubtrain1, ysubtrain1,Xsubtest1,ysubtest1] = subset(1)
[Xsubtrain8, ysubtrain8,Xsubtest8,ysubtest8] = subset(8)

Xsubtrain = np.hstack((Xsubtrain1, Xsubtrain8))
ysubtrain = np.concatenate((ysubtrain1, ysubtrain8))
Xsubtest = np.hstack((Xsubtest1, Xsubtest8))
ysubtest = np.concatenate((ysubtest1, ysubtest8))

#Center X data
Xsubtrain_center = Xsubtrain - np.mean(Xsubtrain,axis = 1, keepdims=True)
Xsubtest_center = Xsubtest - np.mean(Xsubtest,axis = 1, keepdims=True)


#Transform into PC modes
Xsubtrain_pc = dU_trunc.T@Xsubtrain
Xsubtest_pc = dU_trunc.T@Xsubtest

#Do Ridge Classification on 1's and 3's
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import cross_val_score

RidgeCL = RidgeClassifierCV()
RidgeCL.fit(Xsubtrain_pc.T, ysubtrain)

score = cross_val_score(RidgeCL, Xsubtrain_pc.T, ysubtrain, cv=5)
print("Ridge Classifier 1 & 3  Accuracy: {:.4f}".format(RidgeCL.score(Xsubtest_pc.T, ysubtest)))
print("Ridge Classifier 1 & 3  5-Fold Cross Validation: {:.4f} +/- {:.4f}".format(score.mean(), score.std() ))


#Task 5: Repeat the procedure with the following pairs: 3/8 and 2/7

##### 3/8 #####
[Xsubtrain1, ysubtrain1,Xsubtest1,ysubtest1] = subset(3)
[Xsubtrain8, ysubtrain8,Xsubtest8,ysubtest8] = subset(8)

Xsubtrain = np.hstack((Xsubtrain1, Xsubtrain8))
ysubtrain = np.concatenate((ysubtrain1, ysubtrain8))
Xsubtest = np.hstack((Xsubtest1, Xsubtest8))
ysubtest = np.concatenate((ysubtest1, ysubtest8))

#Center X data
Xsubtrain_center = Xsubtrain - np.mean(Xsubtrain,axis = 1, keepdims=True)
Xsubtest_center = Xsubtest - np.mean(Xsubtest,axis = 1, keepdims=True)

#Transform into PC modes
Xsubtrain_pc = dU_trunc.T@Xsubtrain
Xsubtest_pc = dU_trunc.T@Xsubtest

#Ridge Classification for 3 and 8
RidgeCL = RidgeClassifierCV()
RidgeCL.fit(Xsubtrain_pc.T, ysubtrain)

score = cross_val_score(RidgeCL, Xsubtrain_pc.T, ysubtrain, cv=5)
print("Ridge Classifier 3 & 8  Accuracy: {:.4f}".format(RidgeCL.score(Xsubtest_pc.T, ysubtest)))
print("Ridge Classifier 3 & 8  5-Fold Cross Validation: {:.4f} +/- {:.4f}".format(score.mean(), score.std() ))


####### 2 & 7 #######
[Xsubtrain1, ysubtrain1,Xsubtest1,ysubtest1] = subset(2)
[Xsubtrain8, ysubtrain8,Xsubtest8,ysubtest8] = subset(7)

Xsubtrain = np.hstack((Xsubtrain1, Xsubtrain8))
ysubtrain = np.concatenate((ysubtrain1, ysubtrain8))
Xsubtest = np.hstack((Xsubtest1, Xsubtest8))
ysubtest = np.concatenate((ysubtest1, ysubtest8))

#Center X data
Xsubtrain_center = Xsubtrain - np.mean(Xsubtrain,axis = 1, keepdims=True)
Xsubtest_center = Xsubtest - np.mean(Xsubtest,axis = 1, keepdims=True)

#Transform into PC modes
Xsubtrain_pc = dU_trunc.T@Xsubtrain
Xsubtest_pc = dU_trunc.T@Xsubtest

#Ridge Classification for 3 and 8
RidgeCL = RidgeClassifierCV()
RidgeCL.fit(Xsubtrain_pc.T, ysubtrain)

score = cross_val_score(RidgeCL, Xsubtrain_pc.T, ysubtrain, cv=5)
print("Ridge Classifier 2 & 7  Accuracy: {:.4f}".format(RidgeCL.score(Xsubtest_pc.T, ysubtest)))
print("Ridge Classifier 2 & 7  5-Fold Cross Validation: {:.4f} +/- {:.4f}".format(score.mean(), score.std() ))


#TASK 6: Use all the digits and perform multi-class classification with Ridege, KNN, and LDA classifiers
Xtrain_pc = np.diag(ds[0:k])@dVt[0:k,:]
Xtest_pc = dU_trunc.T@(Xtestdata - np.mean(Xtestdata,axis = 1, keepdims=True))

#Ridge:
RidgeCL = RidgeClassifierCV()
RidgeCL.fit(Xtrain_pc.T, ytrainlabels)
print("Ridge Classifier Total Set  Accuracy: {:.4f}".format(RidgeCL.score(Xtest_pc.T, ytestlabels)))

#KNN:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

KNNCL = KNeighborsClassifier(n_neighbors=10)
KNNCL.fit(Xtrain_pc.T, ytrainlabels)
print("KNN Classifier Total Set  Accuracy: {:.4f}".format(KNNCL.score(Xtest_pc.T, ytestlabels)))

#LDA:
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
LDACL = LinearDiscriminantAnalysis()
LDACL.fit(Xtrain_pc.T, ytrainlabels)
print("LDA Classifier Total Set  Accuracy: {:.4f}".format(LDACL.score(Xtest_pc.T, ytestlabels)))