from skorch import NeuralNetBinaryClassifier
from data_loading import HistopathDataset
import skorch.callbacks as scb
from sklearn import metrics
import pickle
import torch
import os


from skorch.utils import get_dim
from skorch.utils import is_dataset
from torch.utils.data import DataLoader


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




def train_model(classifier, train_labels, test_lables, file_dir, transform):
    
    ################
    ## Data Loader
    ################    
    
    ### using new data loader and new data split ###
    # create train and test data sets
    dataset_train = HistopathDataset(
        label_file = os.path.abspath(train_labels),
        root_dir = os.path.abspath(file_dir),
        transform = transform)
    
    dataset_test = HistopathDataset(
        label_file = os.path.abspath(test_lables),
        root_dir = os.path.abspath(file_dir),
        transform = transform)
    
    
    
    ######################
    # Definition of Scoring Methods
    ######################
    
    def test_accuracy(net, X = None, y = None):
        y = [y for _, y in dataset_test]
        y_hat = net.predict(dataset_test)
        return metrics.accuracy_score(y, y_hat)
    
    
    
    classifier.callbacks.extend([
                 ('train_acc', scb.EpochScoring('accuracy',
                                                name='train_acc',
                                                lower_is_better = False,
                                                on_train = True)),
                 ('test_acc', scb.EpochScoring(test_accuracy, 
                                               name = 'test_acc',
                                               lower_is_better = False,
                                               on_train = True,
                                               use_caching = False)), # not sure if caching should be disabled here or not ...                                             
                 scb.ProgressBar()])
    
    ######################
    # Model Training
    ######################
    print('''Starting Training for {} 
          \033[1mModel-Params:\033[0m
              \033[1mCriterion:\033[0m     {}
              \033[1mOptimizer:\033[0m     {}
              \033[1mLearning Rate:\033[0m {}
              \033[1mEpochs:\033[0m        {}
              \033[1mBatch size:\033[0m    {}'''
          .format(classifier.module,
                  classifier.criterion,
                  classifier.optimizer,
                  classifier.lr,
                  classifier.max_epochs,
                  classifier.batch_size))
    
    classifier.fit(X = dataset_train, y = None)  
    
    ######################
    # Model Saving
    ######################
    
    # torch.save(gs.best_estimator_, 'test.pt')
    
    ######################
    # Model Loading
    ######################
    
    # model = torch.load('test.pt')
    # print(model)



