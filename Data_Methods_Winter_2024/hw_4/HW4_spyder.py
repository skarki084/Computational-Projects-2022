# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:35:59 2024

@author: samip
"""
import numpy as np
import torch
from torch import nn
import tqdm

import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

train_dataset = torchvision.datasets.FashionMNIST('data/', train=True, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

test_dataset = torchvision.datasets.FashionMNIST('data/', train=False, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

# Use the following code to create a validation set of 10%
train_indices, val_indices, _, _ = train_test_split(
    range(len(train_dataset)),
    train_dataset.targets,
    stratify=train_dataset.targets,
    test_size=0.1,
)

# Generate training and validation subsets based on indices
train_split = Subset(train_dataset, train_indices)
val_split = Subset(train_dataset, val_indices)


# set batches sizes
train_batch_size = 512 #Define train batch size
test_batch_size  = 256 #Define test batch size (can be larger than train batch size)

# Define dataloader objects that help to iterate over batches and samples for
# training, validation and testing
train_batches = DataLoader(train_split, batch_size=train_batch_size, shuffle=True)
val_batches = DataLoader(val_split, batch_size=train_batch_size, shuffle=True)
test_batches = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)





#Define your (As Cool As It Gets) Fully Connected Neural Network 
class ACAIGFCN(nn.Module):
    #Initialize model layers, add additional arguments to adjust
    def __init__(self, input_dim, output_dim, hidden1_dim): 
        super(ACAIGFCN, self).__init__()
        
        self.layer1 = torch.nn.Linear(input_dim, hidden1_dim)
        self.layer2 = torch.nn.Linear(hidden1_dim, output_dim)
        
#Define the network layer(s) and activation function(s)
    def forward(self, x):
            #Define how your model propagates the input through the network
            out1 = torch.nn.functional.relu(self.layer1(x))
            output = self.layer3(out1)
            
            return output
    

# Initialize neural network model with input, output and hidden layer dimensions
model = ACAIGFCN(input_dim = 784, output_dim = 10, hidden1_dim = 10) #... add more parameters

# Define the learning rate and epochs number
learning_rate = 0.1
epochs = 100

train_loss_list = np.zeros((epochs,))
validation_accuracy_list = np.zeros((epochs,))

# Define loss function  and optimizer
loss_func = torch.nn.CrossEntropyLoss()# Use Cross Entropy loss from torch.nn 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)# Use optimizers from torch.optim

# Iterate over epochs, batches with progress bar and train+ validate the ACAIGFCN
# Track the loss and validation accuracy
for epoch in tqdm.trange(epochs):

    # ACAIGFCN Training 
    for train_features, train_labels in train_batches:
        # Set model into training mode
        model.train()
        
        # Reshape images into a vector
        train_features = train_features.reshape(-1, 28*28)

        # Reset gradients, Calculate training loss on model
        optimizer.zero_grad()
        loss = loss_func(train_features, train_labels)
        # Perfrom optimization, back propagation
        loss.backward()
        
        optimizer.step()
 
    # Record loss for the epoch
    train_loss_list[epoch] = loss.item()
    
    # ACAIGFCN Validation
    for val_features, val_labels in val_batches:
        
        # Telling PyTorch we aren't passing inputs to network for training purpose
        with torch.no_grad(): 
            model.eval()
            
             # Reshape validation images into a vector
            val_features = val_features.reshape(-1, 28*28)
          
            # Compute validation outputs (targets)
            validation_outputs = model(val_features)
            # and compute accuracy 
            correct = (torch.argmax(validation_outputs, dim=1) == 
                   validation_outputs).type(torch.FloatTensor)
            
    # Record accuracy for the epoch; print training loss, validation accuracy
    val_acc = correct.mean()
    num_val_batches = len(val_batches)
    print("Epoch: "+ str(epoch) +"; Validation Accuracy:" + str(val_acc/num_val_batches*100) + '%')


