import torchvision

import architecture
from architecture import LeNet, DenseNet121, ResNet34
from data_loading import HistopathDataset, ToTensor

import numpy as np
from sklearn.datasets import make_classification
from torch import nn
from skorch import NeuralNet, NeuralNetBinaryClassifier
import skorch.callbacks as scb
from torchvision import transforms
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
    label_file = os.path.abspath("data/train_split.csv"),
    root_dir = os.path.abspath("data/train"),
    transform = ToTensor())

dataset_test = HistopathDataset(
    label_file = os.path.abspath("data/test_split.csv"),
    root_dir = os.path.abspath("data/train"),
    transform = ToTensor())

trans_resnet = transforms.Compose([transforms.ToPILImage(),
                                  transforms.Pad(64, padding_mode='reflect'), # 96 + 2*64 = 224
                                  transforms.RandomHorizontalFlip(),  # TODO: model expects normalized channel values (substract means)
                                  transforms.RandomVerticalFlip(),
                                  transforms.RandomRotation(20),
                                  transforms.ToTensor()])

trans_resnet_test = transforms.Compose([transforms.ToPILImage(),
                                  transforms.Pad(64, padding_mode='reflect'),
                                  transforms.ToTensor()])

dataset_train_resnet34 = HistopathDataset(
    label_file = os.path.abspath("data/train_split.csv"),
    root_dir = os.path.abspath("data/train"),
    transform = trans_resnet)

dataset_test_resnet34 = HistopathDataset(
    label_file = os.path.abspath("data/test_split.csv"),
    root_dir = os.path.abspath("data/train"),
    transform = trans_resnet_test)

# print(dataset_train.__getitem__(1))




# TODO define grid search loop
# Option 1: try all combinations
# Option 2: combinations are defined by positions in arrays

# Path to files needs to be definable for easy use in colab 


######################
# Definition of Scoring Methods
######################

def test_accuracy(net, X = None, y = None):
    if net.module == architecture.ResNet34:
        dat = dataset_test_resnet34
    else:
        dat = dataset_test
    y = [y for _, y in dat]
    y_hat = net.predict(dat)
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
    callbacks = [scb.LRScheduler(policy = 'ExponentialLR', gamma = 0.9), # TODO check if this actually works 
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

res_net_34 = NeuralNetBinaryClassifier(
    ResNet34,
    criterion = nn.BCEWithLogitsLoss,
    optimizer = torch.optim.Adam,
    optimizer__weight_decay = 0,
    max_epochs = 6,  # a
    lr = 0.001,
    batch_size = 128,
    iterator_train__shuffle = True,
    train_split = None,
    callbacks = [scb.LRScheduler(policy = 'StepLR', gamma = 0.25, step_size=2),
                 ('train_acc', scb.EpochScoring('accuracy',
                                                name='train_acc',
                                                lower_is_better = False,
                                                on_train = True)),
                 ('test_acc', scb.EpochScoring(test_accuracy,
                                               name = 'test_acc',
                                               lower_is_better = False,
                                               on_train = True,
                                               use_caching = False)),
                 scb.ProgressBar()],
    device ='cuda'
)


######################
# Model Training
######################
print("Starting with model training: ")
# dens_net_121.fit(X = dataset_train, y = None) # TODO print model parameters
res_net_34.fit(X = dataset_train_resnet34, y = None)

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
