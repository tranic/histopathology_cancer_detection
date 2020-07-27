import os
import uuid
import skorch.callbacks as scb
from skorch import NeuralNetBinaryClassifier
import neptune
import torch
from skorch.callbacks.logging import NeptuneLogger
from skorch.dataset import CVSplit

from data_loading import HistopathDataset
from skorch.utils import get_dim
from skorch.utils import is_dataset
from torch.utils.data import DataLoader
from architecture import VGG11, VGG19, DenseNet121, DenseNet201, ResNet18_96, ResNet152_96
import argparse
from torchvision import transforms





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




parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--trainlabels", "-trnl", help="set training label path")
parser.add_argument("--files", "-f", help="set file  path")
parser.add_argument("--output", "-o", help="set output path")
parser.add_argument("--model", "-m", help="specify model")
parser.add_argument("--name", "-n", help="specify neptune name")

args = parser.parse_args()


logger_data = {
                "api_token": "",
                "project_qualified_name": "elangenhan/hcd-experiments",
                "experiment_name": "{} - Normalized - {}".format(args.model, args.name)
            }
    
    ################
    ## Data Loader
    ################    
    
    ### using new data loader and new data split ###
    # create train and test data sets
dataset_train = HistopathDataset(
        label_file = os.path.abspath(args.trainlabels),
        root_dir = os.path.abspath(args.files),
        transform = transforms.Compose([transforms.ToPILImage(),
                                  # transforms.Pad(64, padding_mode='reflect'), # 96 + 2*64 = 224
                                  transforms.RandomHorizontalFlip(),  # TODO: model expects normalized channel values (substract means)
                                  transforms.RandomVerticalFlip(),
                                  transforms.RandomRotation(20),
                                  transforms.ToTensor()]),
        in_memory = True)

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

def parameterized_vgg11():
        return NeuralNetBinaryClassifier(
            VGG11,
            optimizer = torch.optim.Adamax, 
            max_epochs = 30,
            lr = 0.001,
            batch_size = 128,
            iterator_train__shuffle = True, # Shuffle training data on each epoch
            train_split = CVSplit(cv = 0.2, random_state = 42),
            callbacks = callback_list, 
            device ='cuda')
    
def parameterized_vgg19():
        return NeuralNetBinaryClassifier(
            VGG19,
            optimizer = torch.optim.Adamax, 
            max_epochs = 30,
            lr = 0.001,
            batch_size = 128,
            iterator_train__shuffle = True, # Shuffle training data on each epoch
            train_split = CVSplit(cv = 0.2, random_state = 42),
            callbacks = callback_list, 
            device ='cuda')
        
    
def parameterized_resnet18_96():
        return NeuralNetBinaryClassifier(
            ResNet18_96,
            optimizer = torch.optim.Adam, 
            max_epochs = 30,
            lr = 0.01,
            batch_size = 128,
            iterator_train__shuffle = True, # Shuffle training data on each epoch
            train_split = CVSplit(cv = 0.2, random_state = 42),
            callbacks = callback_list, 
            device ='cuda')
    
def parameterized_resnet152_96():
        return NeuralNetBinaryClassifier(
            ResNet152_96,
            optimizer = torch.optim.Adam, 
            max_epochs = 30,
            lr = 0.01,
            batch_size = 128,
            iterator_train__shuffle = True, # Shuffle training data on each epoch
           train_split = CVSplit(cv = 0.2, random_state = 42),
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
            train_split = CVSplit(cv = 0.2, random_state = 42),
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
            train_split = CVSplit(cv = 0.2, random_state = 42),
            callbacks = callback_list, 
            device ='cuda')
    
model_switcher = {'vgg11': parameterized_vgg11,
                  'vgg19': parameterized_vgg19,
                  'densenet121': parameterized_densenet121,
                  'densenet201': parameterized_densenet201,
                  'resnet18_96': parameterized_resnet18_96,
                  'resnet152_96': parameterized_resnet152_96}

     
get_model = model_switcher.get(args.model, lambda: "Model does not exist")

    
classifier = get_model()

params = {}
        
if classifier and classifier.callbacks and classifier.callbacks[0]:
    if hasattr(classifier.callbacks[0], "policy"): params["scheduler_policy"] = classifier.callbacks[0].policy
    if hasattr(classifier.callbacks[0], "step_size"): params["scheduler_step_size"] = classifier.callbacks[0].step_size
    if hasattr(classifier.callbacks[0], "gamma"): params["scheduler_gamma"] = classifier.callbacks[0].gamma

neptune.init(
            api_token=logger_data["api_token"],
            project_qualified_name=logger_data["project_qualified_name"]
        )
experiment = neptune.create_experiment(
            name=logger_data["experiment_name"],
            params={**classifier.get_params(), **params}
        )
logger = NeptuneLogger(experiment, close_after_train=False)

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
    
    
    
    
    
    
    
        
    
   
