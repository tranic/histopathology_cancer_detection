from architecture import LeNet, DenseNet121
from data_loading import HistopathDataset, ToTensor

import numpy as np
from sklearn.datasets import make_classification
from torch import nn
from skorch import NeuralNet, NeuralNetBinaryClassifier
import skorch.callbacks as scb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import metrics
import pickle
import torch
import os


from skorch.utils import get_dim
from skorch.utils import is_dataset
from torch.utils.data import DataLoader

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




# TODO define grid search loop
# Option 1: try all combinations
# Option 2: combinations are defined by positions in arrays

# Path to files needs to be definable for easy use in colab 


######################
# Definition of Scoring Methods
######################

def test_accuracy(net, X = None, y=None):
    y = [y for _, y in dataset_test]
    y_hat = net.predict(dataset_test)
    return metrics.accuracy_score(y, y_hat)


def train_accuracy(net, ds, y=None):
    y = [y for _, y in ds]
    y_hat = net.predict(ds)
    return metrics.accuracy_score(y, y_hat)


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


# this is only temporary monekey patching, we should probably inherit class and just overwrite this single method
def custom_check_data(self, X, y):
        # super().check_data(X, y)
        if (
                (y is None) and
                (not is_dataset(X)) and
                (self.iterator_train is DataLoader)
        ): 
            msg = ("No y-values are given (y=None). You must either supply a "
                   "Dataset as X or implement your own DataLoader for "
                   "training (and your validation) and supply it using the "
                   "``iterator_train`` and ``iterator_valid`` parameters "
                   "respectively.")
            raise ValueError(msg)
        
        if y is not None and get_dim(y) != 1:
            raise ValueError("The target data should be 1-dimensional.")

NeuralNetBinaryClassifier.check_data = custom_check_data


dens_net_121 = NeuralNetBinaryClassifier(
    DenseNet121,
    criterion = nn.BCEWithLogitsLoss, # default can be changed to whatever we need
    optimizer = torch.optim.Adam, 
    optimizer__weight_decay = 0,
    max_epochs = 20,
    lr = 0.01,
    batch_size = 64,
    iterator_train__shuffle = True, # Shuffle training data on each epoch
    train_split = None,
    callbacks = [scb.LRScheduler(policy='ExponentialLR', gamma = 0.9), # TODO check if this actually works 
                 ('train_acc', scb.EpochScoring('accuracy',
                                                name='train_acc',
                                                lower_is_better = False,
                                                on_train = True)),
                 ('test_acc', scb.EpochScoring(test_accuracy, 
                                               name = 'test_acc',
                                               lower_is_better = False,
                                               on_train = True,
                                               use_caching = False)), # not sure if caching should be disabled here or not ...                                             
                 scb.ProgressBar()], 
    device ='cuda'
)


######################
# Model Training
######################
print("Starting with model training: ")
dens_net_121.fit(X = dataset_train, y = None) # TODO print model parameters

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
