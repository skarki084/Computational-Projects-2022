# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 14:14:44 2024

@author: samip
"""

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sklearn.metrics
#plt.style.use('seaborn-poster')



#To start, I will load in all the training data. For each training data set will be
#flattened and made into a column of X_train. Y_train represents the labels, 0 is walking
#1 is jumping, 2 is running

#filename and folder to plot
folder = "hw2data/train/"
file_list = os.listdir(folder)
numpy_files = [file for file in file_list if file.endswith('.npy')]


X_train = np.zeros ((11400,15))
Y_train = np.zeros(15)

Y_train[0:5] = 1
Y_train[5:10] = 2

for i in range(len(numpy_files)):
    path = os.path.join(folder, numpy_files[i])
    array = numpy_array = np.load(path)
    X_train[:,i] = array.flatten()
    
Y_train[0:5] = 1
Y_train[5:10] = 2


#I will take X_train, center it, and take SVD
centered_x_train = X_train - np.mean(X_train, axis=1)[:, None]
dU,ds,dVt = np.linalg.svd(centered_x_train)
dV = dVt.T

#I want to see the Energy of this data, how many principal components do I need
#to approxiamate X_train up to 70%, 80% 90%, 95% of energy? 
E = np.power(ds,2)/np.sum(np.power(ds,2))
cumsum = np.cumsum(E)
plt.plot(cumsum)

plt.title('Cumulative Sum of Energy of Principal Components')
plt.xlabel('index $j$')
plt.ylabel('$\Sigma E_j$')
plt.show()

#70% --> 3 PC nodes
#80% --> 4 PC nodes
#90% --> 6 PC nodes
#95% --> 8 PC nodes


#For each column in centered_x_train, if that is called D, we have:
# D(:,k) = Sum_j ( dU[:.j] * ds[j] * dV(j.k) )
#so, if the columns dU represnt the principle components,
# ds[j] * dV[j,k] represents the coefficient of the jth principle component
#for the first 3 principle components, I will see what the coefficients are for those 
#principle coefficients for each column of the centered_x_train

#after, I will plot these coefficients, which is like seeing each point in the
#PC 0, PC1, PC3 plane 
        
principle_node_components3 = np.diag(ds)[0:3,0:3]@dVt[0:3,:]


#2d plot with PC0 on the x axis and PC1 on the y axis

for i in range(0,15):
    if Y_train[i] == 0:
        color = 'red'#walking
    if Y_train[i]==1:
        color = 'blue' #jumping
    if Y_train[i] == 2: 
        color = 'green' #running
    plt.scatter(principle_node_components3[0,i], principle_node_components3[1,i], c = color)

plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markersize=5, markerfacecolor='red', label='Walking'),
    plt.Line2D([0], [0], marker='o', color='w', markersize=5, markerfacecolor='blue', label='Jumping'),
    plt.Line2D([0], [0], marker='o', color='w', markersize=5, markerfacecolor='green', label='Running')
], loc='upper right')

plt.xlabel('PC0')
plt.ylabel('PC1')
plt.title(r'$X_{train}$ Projected onto first 2 Principle Components')

plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#3D plot with PC0 on x-axis, PC1 on y axis, PC2 on Z axis
for i in range(0,15):
    if Y_train[i] == 0:
        color = 'red'#walking
    if Y_train[i]==1:
        color = 'blue' #jumping
    if Y_train[i] == 2: 
        color = 'green' #running
    ax.scatter(principle_node_components3[0,i], principle_node_components3[1,i], principle_node_components3[2,i], c = color)

legend_location = (1.3, 1)

plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markersize=5, markerfacecolor='red', label='Walking'),
    plt.Line2D([0], [0], marker='o', color='w', markersize=5, markerfacecolor='blue', label='Jumping'),
    plt.Line2D([0], [0], marker='o', color='w', markersize=5, markerfacecolor='green', label='Running')
], bbox_to_anchor = legend_location)

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

ax.set_xlabel('PC0', labelpad = 1)
ax.set_ylabel('PC1', labelpad = 1)
ax.set_zlabel('PC2', labelpad = 1)
plt.title(r'$X_{train}$ Projected onto first 3 Principle Components')

plt.show()


#I will calculate the centroid for each label, Walking, Jumping, and running 
#the centroid will give me the center over all the principal components
#basically, I will get the components of the the principlal components with
#dS @ dVt, then group each column accordingly to what their label is--
#first 500 of them are jumping, second 500 of them are running, last 500 are walking
#when I group them like this, I just need to average each row to get the centroid

p_mode_coefs = np.diag(ds)@dVt

pmcoefs_j = p_mode_coefs[:,0:5]
pmcoefs_r = p_mode_coefs[:,5:10]
pmcoefs_w = p_mode_coefs[:,10:15]

centroid_j = np.mean(pmcoefs_j, axis = 1)
centroid_r = np.mean(pmcoefs_r, axis = 1)
centroid_w = np.mean(pmcoefs_w, axis = 1)

#I computed the centroids for jumping running and walking,
#from the x_train data set


#I am now down to number 5 on the worksheet. I am going to make a "function" which predicts
#what the labels are. Here is how I am doing the prediction--

#I for each column in the p_mod_coefs, which represents X_train in the PCA basis
#I will find the distance between each the X_train observation (in PCA space) and each of the centroids
#which ever has a the lowest. I will do this for various number of PC modes, k, and see how my accuracy changes as I increase k

def classifier(X_train, k):
    #k is the number of PC nodes I want
    #X_train is in PC node basis
    #This will output a vector of guesses, which has the same size as the number of columns of X_train input
    #Y_guess will have values of 0 1 2 for walk, jump, and run guess
    X_train = X_train[0:k,:]
    
    Y_guess = np.zeros(X_train.shape[1])
    
    for i in range(len(Y_guess)):
        j_score = np.linalg.norm(X_train[:,i] - centroid_j[0:k])
        r_score = np.linalg.norm(X_train[:,i] - centroid_r[0:k])
        w_score = np.linalg.norm(X_train[:,i] - centroid_w[0:k])
        
        scores = np.array([w_score, j_score,r_score])
        
        Y_guess[i] = np.argmin(scores)

    return Y_guess

#Now, I take different amounts of k, and I will see how my accuracy of my classifier changes
#I will still be using X_train and Y_train, whis is the training dataset
X_train_PC = p_mode_coefs

train_acc_list = np.zeros(15)
for k in range(1,15):
    Y_guess = classifier(X_train_PC, k)
    acc = sklearn.metrics.accuracy_score(Y_train, Y_guess)
    train_acc_list[k] = acc

plt.plot(train_acc_list*100)
plt.xlabel('k- number of PC modes')
plt.ylabel('Accuracy (%)')
plt.title(r'Classifier Accuracy with Training Dataset')
plt.show()
#Now, I will load in a new testing dataset, I will use the same classifier with that testing dataset.
folder = "hw2data/test/"
file_list = os.listdir(folder)
numpy_files = [file for file in file_list if file.endswith('.npy')]


X_test = np.zeros ((11400,3))
Y_test = np.zeros(3)

Y_test[0] = 1
Y_test[1] = 2
Y_test[2] = 0
for i in range(len(numpy_files)):
    path = os.path.join(folder, numpy_files[i])
    array = numpy_array = np.load(path)
    X_test[:,i] = array.flatten()

#I don't want to re-compute SVD, I want to take this new X_test, and go to the basis that I made the classifier in
#dU.T will change X_test into the PC mode basis
X_test_PC = (dU.T@(X_test - np.mean(X_t, axis=1)[:, None]))[0:15,:] 
#I want only the first 15 rows, because other rows are not needed. I only use maximum of 15 principal components

#I will run the classifier again on these, for different values of k

test_acc_list = np.zeros(15)
for k in range(1,15):
    Y_guess = classifier(X_test_PC, k)
    acc = sklearn.metrics.accuracy_score(Y_test, Y_guess)
    test_acc_list[k] = acc

plt.plot(test_acc_list*100)
plt.xlabel('k- number of PC modes')
plt.ylabel('Accuracy (%)')
plt.title(r'Classifier Accuracy with Testing Dataset')
plt.show()