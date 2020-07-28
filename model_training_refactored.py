import os
import uuid
import skorch.callbacks as scb
import neptune
from skorch.callbacks.logging import NeptuneLogger
import pandas as pd
from data_loading import HistopathDataset
import torch
from skorch.helper import predefined_split
from sklearn import metrics


def train_model(classifier, train_labels, test_labels, file_dir, train_transform, test_transform, in_memory, output_path, logger = None):


    if logger:  
        params = {}
        
        if classifier and classifier.callbacks and classifier.callbacks[0]:
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
        transform = train_transform,
        in_memory = in_memory)
    
    dataset_test = HistopathDataset(
        label_file = os.path.abspath(test_labels),
        root_dir = os.path.abspath(file_dir),
        transform = test_transform,
        in_memory = in_memory)


     ######################
    # Definition of Scoring Methods
    ######################
    
    def test_roc_auc(net, ds, y = None):
        y_hat = net.predict_proba(df)
        y_true = [y for _, y in ds]
        return metrics.roc_auc_score(y_true, y_hat[:, 1])   
    
    def test_roc_auc_2(net, X = None, y = None):
        y_pred = net.predict_proba(dataset_test)
        y_true = [y for _, y in dataset_test]
        return metrics.roc_auc_score(y_true, y_pred[:, 1]) 

              
    # Test if scorings are already attached
    classifier.callbacks.extend([
                 ('train_acc', scb.EpochScoring('accuracy',
                                                name='train_acc',
                                                lower_is_better = False,
                                                on_train = True)),
                ('train_f1', scb.EpochScoring('f1',
                                                name='train_f1',
                                                lower_is_better = False,
                                                on_train = True)),
                ('train_roc_auc', scb.EpochScoring('roc_auc',
                                                name='train_roc_auc',
                                                lower_is_better = False,
                                                on_train = True)),
                ('train_precision', scb.EpochScoring('precision',
                                                name='train_precision',
                                                lower_is_better = False,
                                                on_train = True)),
                ('train_recall', scb.EpochScoring('recall',
                                                name='train_recall',
                                                lower_is_better = False,
                                                on_train = True)),
                ('valid_f1', scb.EpochScoring('f1',
                                                name='valid_f1',
                                                lower_is_better = False)),
                ('valid_roc_auc', scb.EpochScoring('roc_auc',
                                                name='valid_roc_auc',
                                                lower_is_better = False)),
                ('valid_roc_auc_test', scb.EpochScoring(test_roc_auc,
                                                name='valid_roc_auc_test',
                                                lower_is_better = False,
                                                use_caching = False)),
                ('valid_roc_auc_test2', scb.EpochScoring(test_roc_auc_2,
                                                name='valid_roc_auc_test2',
                                                lower_is_better = False,
                                                use_caching = False)),
                ('valid_precision', scb.EpochScoring('precision',
                                                name='valid_precision',
                                                lower_is_better = False)),
                ('valid_recall', scb.EpochScoring('recall',
                                                name='valid_recall',
                                                lower_is_better = False)),
                 scb.ProgressBar()])

    
    classifier.train_split = predefined_split(dataset_test)
    
    
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
    
                                     
    df = pd.read_csv(train_labels)
    target = df["label"]                    
    classifier.fit(X = dataset_train, y = torch.Tensor(target))
    
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
        logger.experiment.log_text('test_name', train_labels)
        logger.experiment.log_artifact('{}/{}-model.pkl'.format(output_path, uid))
        logger.experiment.log_artifact('{}/{}-opt.pkl'.format(output_path, uid))
        logger.experiment.log_artifact('{}/{}-history.json'.format(output_path, uid))
        logger.experiment.stop()


    print("Saving completed...")
