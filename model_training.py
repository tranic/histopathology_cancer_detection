import os
import uuid
import skorch.callbacks as scb
from skorch import NeuralNetBinaryClassifier
from sklearn import metrics
import neptune
from skorch.callbacks.logging import NeptuneLogger

from data_loading import HistopathDataset
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


def train_model(classifier, train_labels, test_lables, file_dir, transform, in_memory, output_path, logger = None):


    if logger:  
        params = {}
        
        if hasattr(classifier.callbacks[0], "policy"): params["scheduler_policy"] = classifier.callbacks[0].policy
        if hasattr(classifier.callbacks[0], "step_size"): params["scheduler_step_size"] = classifier.callbacks[0].step_size
        if hasattr(classifier.callbacks[0], "gamma"): params["scheduler_gamma"] = classifier.callbacks[0].gamma

        neptune.init(
            api_token=logger["api_token"],
            project_qualified_name=logger["project_qualified_name"]
        )
        experiment = neptune.create_experiment(
            name=logger["experiment_name"],
            params={**classifier.get_params(), **params}
        )
        logger = NeptuneLogger(experiment, close_after_train=False)

    
    ################
    ## Data Loader
    ################    
    
    ### using new data loader and new data split ###
    # create train and test data sets
    dataset_train = HistopathDataset(
        label_file = os.path.abspath(train_labels),
        root_dir = os.path.abspath(file_dir),
        transform = transform,
        in_memory = in_memory)
    
    dataset_test = HistopathDataset(
        label_file = os.path.abspath(test_lables),
        root_dir = os.path.abspath(file_dir),
        transform = transform,
        in_memory = in_memory)
    
    
    ######################
    # Definition of Scoring Methods
    ######################
    
    def test_accuracy(net, X = None, y = None):
        y = [y for _, y in dataset_test]
        y_hat = net.predict(dataset_test)
        return metrics.accuracy_score(y, y_hat)

    
    # Test if scorings are already attached
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
                ('f1', scb.EpochScoring('f1',
                                                name='f1',
                                                lower_is_better = False,
                                                on_train = True)),
                ('roc_auc', scb.EpochScoring('roc_auc',
                                                name='roc_auc',
                                                lower_is_better = False,
                                                on_train = True)),
                ('precision', scb.EpochScoring('precision',
                                                name='precision',
                                                lower_is_better = False,
                                                on_train = True)),
                ('recall', scb.EpochScoring('recall',
                                                name='recall',
                                                lower_is_better = False,
                                                on_train = True)),
                 scb.ProgressBar()])

    if logger:
        classifier.callbacks.append(logger)
    
    ######################
    # Model Training
    ######################
    print('''Starting Training for {} 
          \033[1mModel-Params:\033[0m
              \033[1mCriterion:\033[0m     {}
              \033[1mOptimizer:\033[0m     {}
              \033[1mLearning Rate:\033[0m {}
              \033[1mEpochs:\033[0m        {}
              \033[1mBatch size:\033[0m    {}
              '''
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
    
    # TODO save this id including the model params for easy retrival and loading
    
    print("Saving model...")
    
    uid = uuid.uuid4()

    classifier.save_params(f_params = '{}/{}-model.pkl'.format(output_path, uid), 
                           f_optimizer='{}/{}-opt.pkl'.format(output_path, uid), 
                           f_history='{}/{}-history.json'.format(output_path, uid))

    if logger:
        logger.experiment.log_text('uid', str(uid))
        logger.experiment.log_artifact('{}/{}-model.pkl'.format(output_path, uid))
        logger.experiment.log_artifact('{}/{}-opt.pkl'.format(output_path, uid))
        logger.experiment.log_artifact('{}/{}-history.json'.format(output_path, uid))
        logger.experiment.stop()


    print("Saving completed...")
