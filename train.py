from architecture import LeNet, DensNet121
from data_loading import HistopathDataset, ToTensor

import numpy as np
from sklearn.datasets import make_classification
from torch import nn
from skorch import NeuralNet
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import metrics
import pickle
import torch
import os

################
## Data Loader
################    

### using new data loader and new data split ###
# create train and test data sets
dataset_train = HistopathDataset(
    label_file=os.path.abspath("data/train_split.csv"),
    root_dir=os.path.abspath("data/train"),
    transform=ToTensor())

dataset_test = HistopathDataset(
    label_file=os.path.abspath("data/test_split.csv"),
    root_dir=os.path.abspath("data/train"),
    transform=ToTensor())

# print(dataset_train.__getitem__(1))

######################
# Definition of Net(s)
######################

le_net = NeuralNet(
    LeNet,
    criterion = nn.NLLLoss, # default can be changed to whatever we need
    optimizer = torch.optim.SGD, # default can be changed to whatever we need
    max_epochs = 2,
    lr = 0.1,
    batch_size = 128,
    iterator_train__shuffle = True, # Shuffle training data on each epoch
    train_split = None,
    callbacks = None, # build custom callback for plotting of loss/ accuracy, etc 
    # <-- not sure how well this integrates with the d2l plotting thing though
    # look at current on epoch end implementation to not lose fancy table output!
    device ='cpu'
)


dens_net_121 = NeuralNet(
    DensNet121,
    criterion = nn.BCELoss, # default can be changed to whatever we need
    optimizer = torch.optim.Adam,
    optimizer__weight_decay = 0,
    max_epochs = 2,
    lr = 0.01,
    batch_size = 128,
    iterator_train__shuffle = True, # Shuffle training data on each epoch
    train_split = None,
    callbacks = None, # build custom callback for plotting of loss/ accuracy, etc 
    # <-- not sure how well this integrates with the d2l plotting thing though
    # look at current on epoch end implementation to not lose fancy table output!
    device ='cpu'
)


######################
# Model Training
######################

dens_net_121.fit(dataset_train)
# print("Model-Params: {}".format(net.get_params()))

# doesn't work, have to implement this ourselves 
# print("Mean test accuracy: {}".format(net.score(dataset_test)))


######################
# Model Saving
######################

# torch.save(gs.best_estimator_, 'test.pt')

######################
# Model Loading
######################

# model = torch.load('test.pt')
# print(model)


