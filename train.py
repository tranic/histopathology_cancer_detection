import os
import uuid
import argparse
import neptune
import pandas as pd

import skorch.callbacks as scb
from skorch import NeuralNetBinaryClassifier
from skorch.helper import predefined_split
from skorch.callbacks.logging import NeptuneLogger

import torch
from torchvision import transforms

from data_loading import HistopathDataset
from architecture import VGG11, VGG19, DenseNet121, DenseNet201, ResNet18_96, ResNet152_96, LeNet


######################################
#        COMMAND LINE PARAMS        #
#####################################

parser = argparse.ArgumentParser(description='Set necessary values to train different types of predefined models')
parser.add_argument("--trainlabels", "-trnl", help="set training label path")
parser.add_argument("--testlabels", "-tstl", help="set test label path")
parser.add_argument("--files", "-f", help="set file  path")
parser.add_argument("--output", "-o", help="set output path")
parser.add_argument("--model", "-m", help="specify model")
parser.add_argument("--name", "-n", help="specify neptune name")

parser.add_argument("--apitoken", "-api", help="specify neptune api token")
parser.add_argument("--experiment", "-e", help="specify neptune experiment")

args = parser.parse_args()


logger_data = {
                "api_token": "{}".format(args.apitoken),
                "project_qualified_name": "{}".format(args.expertiment),
                "experiment_name": "{} - {}".format(args.model, args.name)
            }
    
######################################
#            DATA LOADING           #
#####################################
# Create train and test dataset with custom Dataset based on paths supplied as command line args

dataset_train = HistopathDataset(
        label_file = os.path.abspath(args.trainlabels),
        root_dir = os.path.abspath(args.files),
        transform = transforms.Compose([transforms.ToPILImage(),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),
                                  transforms.RandomRotation(20),
                                  transforms.ToTensor()]),
        in_memory = False)

dataset_test = HistopathDataset(
        label_file = os.path.abspath(args.testlabels),
        root_dir = os.path.abspath(args.files),
        transform = transforms.ToTensor(),
        in_memory = False)
    
    
######################################
#          METRIC CALLBACKS         #
#####################################
# We collect the same metrics for both test and train data for all models 
# Also the same learning rate sheduler is used between all runs

# For test metrics "valid" is used as name to be conform with the skorch naming scheme
callback_list = [scb.LRScheduler(policy = 'ExponentialLR', gamma = 0.9),
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
                ('valid_precision', scb.EpochScoring('precision',
                                                name='valid_precision',
                                                lower_is_better = False)),
                ('valid_recall', scb.EpochScoring('recall',
                                                name='valid_recall',
                                                lower_is_better = False)),
                 scb.ProgressBar()]       

######################################
#        CLASSIFIER DEFINITION      #
#####################################
# Each Classifier is initalized with the chosen hyperparameters
# Options are predefined to ensure same hyperparameters through all runs

def parameterized_vgg11():
        return NeuralNetBinaryClassifier(
            VGG11,
            optimizer = torch.optim.Adamax, 
            max_epochs = 30,
            lr = 0.001,
            batch_size = 128,
            iterator_train__shuffle = True,
            train_split = predefined_split(dataset_test), # Supply the skorch framework with our own predefined test dataset
            callbacks = callback_list, 
            device ='cuda')
    
def parameterized_vgg19():
        return NeuralNetBinaryClassifier(
            VGG19,
            optimizer = torch.optim.Adamax, 
            max_epochs = 30,
            lr = 0.001,
            batch_size = 128,
            iterator_train__shuffle = True,
            train_split = predefined_split(dataset_test),
            callbacks = callback_list, 
            device ='cuda')
    
        
def parameterized_resnet18_96():
        return NeuralNetBinaryClassifier(
            ResNet18_96,
            optimizer = torch.optim.Adam, 
            max_epochs = 30,
            lr = 0.01,
            batch_size = 128,
            iterator_train__shuffle = True, 
            train_split = predefined_split(dataset_test),
            callbacks = callback_list, 
            device ='cuda')
    
def parameterized_resnet152_96():
        return NeuralNetBinaryClassifier(
            ResNet152_96,
            optimizer = torch.optim.Adam, 
            max_epochs = 30,
            lr = 0.01,
            batch_size = 128,
            iterator_train__shuffle = True, 
            train_split = predefined_split(dataset_test),
            callbacks = callback_list, 
            device ='cuda')    
    
def parameterized_densenet121():
        return NeuralNetBinaryClassifier(
            DenseNet121,
            optimizer = torch.optim.Adam, 
            max_epochs = 30,
            lr = 0.01,
            batch_size = 128,
            iterator_train__shuffle = True, # Shuffle training data on each epoch
            train_split = predefined_split(dataset_test),
            callbacks = callback_list, 
            device ='cuda')
    
def parameterized_densenet201():
        return NeuralNetBinaryClassifier(
            DenseNet201,
            optimizer = torch.optim.Adam, 
            max_epochs = 30,
            lr = 0.01,
            batch_size = 128,
            iterator_train__shuffle = True, # Shuffle training data on each epoch
            train_split = predefined_split(dataset_test),
            callbacks = callback_list, 
            device ='cuda')
    
def parameterized_lenet():
        return NeuralNetBinaryClassifier(
            LeNet,
            optimizer = torch.optim.Adam, 
            max_epochs = 100,
            lr = 0.01,
            batch_size = 128,
            iterator_train__shuffle = True, # Shuffle training data on each epoch
            train_split = predefined_split(dataset_test),
            callbacks = callback_list, 
            device ='cuda')
    
# Select model based on command line input    
model_switcher = {'vgg11': parameterized_vgg11,
                  'vgg19': parameterized_vgg19,
                  'densenet121': parameterized_densenet121,
                  'densenet201': parameterized_densenet201,
                  'resnet18_96': parameterized_resnet18_96,
                  'resnet152_96': parameterized_resnet152_96,
                  'lenet': parameterized_lenet}

     
get_model = model_switcher.get(args.model, lambda: "Model does not exist")

    
classifier = get_model()

# If available collect additional experiment metrics      

params = {}
 
if classifier and classifier.callbacks and classifier.callbacks[0]:
    if hasattr(classifier.callbacks[0], "policy"): params["scheduler_policy"] = classifier.callbacks[0].policy
    if hasattr(classifier.callbacks[0], "step_size"): params["scheduler_step_size"] = classifier.callbacks[0].step_size
    if hasattr(classifier.callbacks[0], "gamma"): params["scheduler_gamma"] = classifier.callbacks[0].gamma


######################################
#        NEPTUNE SETUP              #
#####################################
# Specifies neptune experiment to collect all metrics across all model training runs

neptune.init(api_token = logger_data["api_token"],
             project_qualified_name = logger_data["project_qualified_name"])

experiment = neptune.create_experiment(name=logger_data["experiment_name"],
                                       params={**classifier.get_params(), **params})

logger = NeptuneLogger(experiment, close_after_train=False)

classifier.callbacks.append(logger)

######################################
#           MODEL TRAINING          #
#####################################
# Print most important hyperparameters at the beginning of the model training

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
    
# Extract training labels from file instead of dataset for increased speed
df = pd.read_csv(args.trainlabels)
target = df["label"]       

# Train the classifier with the fitting function supplied by skorch
# During each epoch the defined callbacks are called to collect metrics              
classifier.fit(X = dataset_train, y = torch.Tensor(target))
 
   
######################################
#           MODEL SAVING            #
#####################################
# Saving the model and its history locally and then upload the artifacts to neptune

print("Saving model...")
  
uid = uuid.uuid4()

classifier.save_params(f_params = '{}/{}-model.pkl'.format(args.output, uid), 
                           f_optimizer='{}/{}-opt.pkl'.format(args.output, uid), 
                           f_history='{}/{}-history.json'.format(args.output, uid))


logger.experiment.log_text('uid', str(uid))
logger.experiment.log_text('test_name', args.trainlabels)
logger.experiment.log_artifact('{}/{}-model.pkl'.format(args.output, uid))
logger.experiment.log_artifact('{}/{}-opt.pkl'.format(args.output, uid))
logger.experiment.log_artifact('{}/{}-history.json'.format(args.output, uid))
logger.experiment.stop()

print("Saving completed...")
