
from test import *
from architecture import MyModule

import numpy as np
from sklearn.datasets import make_classification
from torch import nn
from skorch import NeuralNetClassifier
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import metrics
import pickle
import torch

################
## Data Loader
################

X, y = make_classification(1000, 20, n_informative=10, random_state=0)
X = X.astype(np.float32)
y = y.astype(np.int64)

######################
# Definition of Net(s)
######################

net = NeuralNetClassifier(
    MyModule,
    max_epochs=10,
    lr=0.1,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
    device='cuda'
)

######################
# Pipeline-Definition
######################

net.set_params(train_split=False, verbose=0)
params = {
    'lr': [0.01, 0.02],
    'max_epochs': [10],
    'module__num_units': [30, 50],
}

######################
# Randomized Search
######################


gs = RandomizedSearchCV(net, params, refit=True, cv=3, scoring='accuracy', verbose=0, n_jobs=4)
gs.fit(X, y)
print("Best score: {:.3f}, Best params: {}".format(gs.best_score_, gs.best_params_))
print(gs.best_params_)


######################
# Model Saving
######################

torch.save(gs.best_estimator_, 'test.pt')

######################
# Model Loading
######################

model = torch.load('test.pt')
print(model)


