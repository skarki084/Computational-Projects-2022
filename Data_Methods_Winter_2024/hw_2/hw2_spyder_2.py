# -*- coding: utf-8 -*-

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sklearn.metrics

folder = "hw2data/train/"
file_list = os.listdir(folder)
numpy_files = [file for file in file_list if file.endswith('.npy')]

X_train = np.zeros((114,1500))
Y_train = np.zeros(1500)

for i in range(len(numpy_files)):
    path = os.path.join(folder, numpy_files[i])
    array = numpy_array = np.load(path)
    X_train[:,i*100:(i+1)*100] = array
   
Y_train[0:500] = 1
Y_train[500:1000] = 2


#I will take X_train, center it, and take SVD
centered_x_train = X_train - np.mean(X_train, axis=1)[:, None]
dU,ds,dVt = np.linalg.svd(centered_x_train)
dV = dVt.T

#I want to see the Energy of this data, how many principal components do I need
#to approxiamate X_train up to 70%, 80% 90%, 95% of energy?
E = np.power(ds,2)/np.sum(np.power(ds,2))
cumsum = np.cumsum(E)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.plot(cumsum[0:50])

plt.title('Cumulative Sum of Energy of Principal Components', fontsize = 20)
plt.xlabel('index $j$', fontsize = 15)
plt.ylabel('$\Sigma E_j$' ,fontsize = 15)
plt.show()


#70% --> 1 PC nodes
#80% --> 2 PC nodes
#90% --> 4 PC nodes
#95% --> 6 PC nodes

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

plt.figure(figsize = (8,6))

for i in range(0,1500):
    if Y_train[i] == 0:
        color = 'red'#walking
    if Y_train[i]==1:
        color = 'blue' #jumping
    if Y_train[i] == 2: 
        color = 'green' #running
    plt.scatter(principle_node_components3[0,i], principle_node_components3[1,i], c = color)

#plot centroid
plt.scatter(np.mean(principle_node_components3[0, 0:500]), np.mean(principle_node_components3[1, 0:500]), c='white', marker='*', s=300, edgecolors='black', linewidths=1.5)
plt.scatter(np.mean(principle_node_components3[0, 500:1000]), np.mean(principle_node_components3[1, 500:1000]), c='white', marker='*', s=300, edgecolors='black', linewidths=1.5)
plt.scatter(np.mean(principle_node_components3[0, 1000:1500]), np.mean(principle_node_components3[1, 1000:1500]), c='white', marker='*', s=300, edgecolors='black', linewidths=1.5)

plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor='red', label='Walking'),
    plt.Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor='blue', label='Jumping'),
    plt.Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor='green', label='Running'),
    plt.Line2D([0], [0], marker='*', color='w', markersize=10, markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5, label='Centroid')
], loc='upper right', fontsize = 15)


plt.xlabel('PC0', fontsize = 15)
plt.ylabel('PC1',fontsize = 15)
plt.title(r'$X_{train}$ Projected onto first 2 Principle Components',fontsize = 20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


fig = plt.figure(figsize= (8,6))
ax = fig.add_subplot(111, projection='3d')

#3D plot with PC0 on x-axis, PC1 on y axis, PC2 on Z axis
for i in range(0,1500):
    if Y_train[i] == 0:
        color = 'red'#walking
    if Y_train[i]==1:
        color = 'blue' #jumping
    if Y_train[i] == 2: 
        color = 'green' #running
    ax.scatter(principle_node_components3[0,i], principle_node_components3[1,i], principle_node_components3[2,i], c = color)

#plot centroid
# Plotting the centroids with stars
ax.scatter(np.mean(principle_node_components3[0, 0:500]), np.mean(principle_node_components3[1, 0:500]), 
           np.mean(principle_node_components3[2, 0:500]), c='white', marker='*', s=300, edgecolors='black', linewidths=1.5)
ax.scatter(np.mean(principle_node_components3[0, 500:1000]), np.mean(principle_node_components3[1, 500:1000]), 
           np.mean(principle_node_components3[2, 500:1000]), c='white', marker='*', s=300, edgecolors='black', linewidths=1.5)
ax.scatter(np.mean(principle_node_components3[0, 1000:1500]), np.mean(principle_node_components3[1, 1000:1500]),
           np.mean(principle_node_components3[2, 1000:1500]), c='white', marker='*', s=300, edgecolors='black', linewidths=1.5)

legend_location = (1.3, 1)



plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor='red', label='Walking'),
    plt.Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor='blue', label='Jumping'),
    plt.Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor='green', label='Running'),
    plt.Line2D([0], [0], marker='*', color='w', markersize=10, markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5, label='Centroid')
], bbox_to_anchor = legend_location, fontsize = 15)

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

ax.set_xlabel('PC0', labelpad = 1, fontsize = 15)
ax.set_ylabel('PC1', labelpad = 1, fontsize = 15)
ax.set_zlabel('PC2', labelpad = 1, fontsize = 15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title(r'$X_{train}$ Projected onto first 3 Principle Components', fontsize = 20)

plt.show()

#I will calculate the centroid for each label, Walking, Jumping, and running 
#the centroid will give me the center over all the principal components
#basically, I will get the components of the the principlal components with
#dS @ dVt, then group each column accordingly to what their label is--
#first 500 of them are jumping, second 500 of them are running, last 500 are walking
#when I group them like this, I just need to average each row to get the centroid

p_mode_coefs = np.diag(ds)@dVt[0:114,:]

pmcoefs_j = p_mode_coefs[:,0:500]
pmcoefs_r = p_mode_coefs[:,500:1000]
pmcoefs_w = p_mode_coefs[:,1000:1500]

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

train_acc_list = np.zeros(30)
for k in range(1,30):
    Y_guess = classifier(X_train_PC, k)
    acc = sklearn.metrics.accuracy_score(Y_train, Y_guess)
    train_acc_list[k] = acc

plt.plot(train_acc_list*100)
plt.xlabel('k- number of PC modes', fontsize = 15)
plt.ylabel('Accuracy (%)', fontsize = 15)
plt.title(r'Classifier Accuracy with Training Dataset', fontsize = 20)
plt.ylim([0,105])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


#Now, I will load in a new testing dataset, I will use the same classifier with that testing dataset.
folder = "hw2data/test/"
file_list = os.listdir(folder)
numpy_files = [file for file in file_list if file.endswith('.npy')]


X_test = np.zeros ((114,300))
Y_test = np.zeros(300)

Y_test[0:100] = 1
Y_test[100:200] = 2
Y_test[200:300] = 0
for i in range(len(numpy_files)):
    path = os.path.join(folder, numpy_files[i])
    array = numpy_array = np.load(path)
    X_test[:,i*100:(i+1)*100] = array
    
#I don't want to re-compute SVD, I want to take this new X_test, and go to the basis that I made the classifier in
#dU.T will change X_test into the PC mode basis
X_test_PC = (dU.T[0:114,:]@(X_test - np.mean(X_train, axis=1)[:, None]))
 #I want only the first 114 rows, because other rows are not needed. I only use maximum of 15 principal components

 #I will run the classifier again on these, for different values of k

test_acc_list = np.zeros(30)
for k in range(1,30):
    Y_guess = classifier(X_test_PC, k)
    acc = sklearn.metrics.accuracy_score(Y_test, Y_guess)
    test_acc_list[k] = acc

plt.plot(test_acc_list*100)
plt.xlabel('k- number of PC modes', fontsize = 15)
plt.ylabel('Accuracy (%)', fontsize = 15)
plt.title(r'Classifier Accuracy with Testing Dataset', fontsize = 20)
plt.ylim([0,105])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()