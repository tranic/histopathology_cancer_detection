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
from architecture import ResNet34Pretrained, ResNet152Pretrained, DenseNet121Pretrained, DenseNet201Pretrained


######################################
#        COMMAND LINE PARAMS        #
#####################################

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--trainlabels", "-trnl", help="set training label path")
parser.add_argument("--testlabels", "-tstl", help="set test label path")
parser.add_argument("--files", "-f", help="set file  path")
parser.add_argument("--output", "-o", help="set output path")
parser.add_argument("--model", "-m", help="specify model")
parser.add_argument("--name", "-n", help="specify neptune name")

args = parser.parse_args()

logger_data = {
    "api_token": "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiODIzOTFlNTEtYmIwNi00NDZiLTgyMjgtOGQ5MTllMDU2ZDVlIn0=",
    "project_qualified_name": "elangenhan/hcd-experiments",
    "experiment_name": "{} - Pretrained - {}".format(args.model, args.name)
}

######################################
#            DATA LOADING           #
#####################################

dataset_train = HistopathDataset(
    label_file=os.path.abspath(args.trainlabels),
    root_dir=os.path.abspath(args.files),
    transform=transforms.Compose([transforms.ToPILImage(),
                                  transforms.Pad(64, padding_mode='reflect'),  # 96 + 2*64 = 224
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),
                                  transforms.RandomRotation(20),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.70017236, 0.5436771, 0.6961061],
                                                       std=[0.22246036, 0.26757348, 0.19798167])]),
    in_memory=False)

dataset_test = HistopathDataset(
    label_file=os.path.abspath(args.testlabels),
    root_dir=os.path.abspath(args.files),
    transform=transforms.ToTensor(),
    in_memory=False)


######################################
#          METRIC CALLBACKS         #
#####################################

callback_list = [scb.LRScheduler(policy='StepLR', gamma=0.4, step_size=2),
                 ('train_acc', scb.EpochScoring('accuracy',
                                                name='train_acc',
                                                lower_is_better=False,
                                                on_train=True)),
                 ('train_f1', scb.EpochScoring('f1',
                                               name='train_f1',
                                               lower_is_better=False,
                                               on_train=True)),
                 ('train_roc_auc', scb.EpochScoring('roc_auc',
                                                    name='train_roc_auc',
                                                    lower_is_better=False,
                                                    on_train=True)),
                 ('train_precision', scb.EpochScoring('precision',
                                                      name='train_precision',
                                                      lower_is_better=False,
                                                      on_train=True)),
                 ('train_recall', scb.EpochScoring('recall',
                                                   name='train_recall',
                                                   lower_is_better=False,
                                                   on_train=True)),
                 ('valid_f1', scb.EpochScoring('f1',
                                               name='valid_f1',
                                               lower_is_better=False)),
                 ('valid_roc_auc', scb.EpochScoring('roc_auc',
                                                    name='valid_roc_auc',
                                                    lower_is_better=False)),
                 ('valid_precision', scb.EpochScoring('precision',
                                                      name='valid_precision',
                                                      lower_is_better=False)),
                 ('valid_recall', scb.EpochScoring('recall',
                                                   name='valid_recall',
                                                   lower_is_better=False)),
                 scb.ProgressBar()]


def parameterized_resnet34():
    return NeuralNetBinaryClassifier(
        ResNet34Pretrained,
        optimizer=torch.optim.Adam,
        max_epochs=30,
        lr=0.001,
        batch_size=128,
        iterator_train__shuffle=True,  # Shuffle training data on each epoch
        train_split=predefined_split(dataset_test),
        callbacks=callback_list,
        device='cuda')


def parameterized_resnet152():
    return NeuralNetBinaryClassifier(
        ResNet152Pretrained,
        optimizer=torch.optim.Adam,
        max_epochs=30,
        lr=0.001,
        batch_size=128,
        iterator_train__shuffle=True,  
        train_split=predefined_split(dataset_test),
        callbacks=callback_list,
        device='cuda')


def parameterized_densenet121():
    return NeuralNetBinaryClassifier(
        DenseNet121Pretrained,
        optimizer=torch.optim.Adam,
        max_epochs=30,
        lr=0.01,
        batch_size=128,
        iterator_train__shuffle=True,
        train_split=predefined_split(dataset_test),
        callbacks=callback_list,
        device='cuda')


def parameterized_densenet201():
    return NeuralNetBinaryClassifier(
        DenseNet201Pretrained,
        optimizer=torch.optim.Adam,
        max_epochs=30,
        lr=0.01,
        batch_size=128,
        iterator_train__shuffle=True,
        train_split=predefined_split(dataset_test),
        callbacks=callback_list,
        device='cuda')


######################################
#        CLASSIFIER DEFINITION      #
#####################################

model_switcher = {'densenet121': parameterized_densenet121,
                  'densenet201': parameterized_densenet201,
                  'resnet34': parameterized_resnet34,
                  'resnet152': parameterized_resnet152}

get_model = model_switcher.get(args.model, lambda: "Model does not exist")

classifier = get_model()


######################################
#        NEPTUNE SETUP              #
#####################################

params = {}

if classifier and classifier.callbacks and classifier.callbacks[0]:
    if hasattr(classifier.callbacks[0], "policy"): params["scheduler_policy"] = classifier.callbacks[0].policy
    if hasattr(classifier.callbacks[0], "step_size"): params["scheduler_step_size"] = classifier.callbacks[0].step_size
    if hasattr(classifier.callbacks[0], "gamma"): params["scheduler_gamma"] = classifier.callbacks[0].gamma

neptune.init(api_token=logger_data["api_token"],
             project_qualified_name=logger_data["project_qualified_name"])

experiment = neptune.create_experiment(name=logger_data["experiment_name"],
                                       params={**classifier.get_params(), **params})

logger = NeptuneLogger(experiment, close_after_train=False)

classifier.callbacks.append(logger)

######################################
#           MODEL TRAINING          #
#####################################

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

df = pd.read_csv(args.trainlabels)
target = df["label"]
classifier.fit(X=dataset_train, y=torch.Tensor(target))

######################################
#           MODEL SAVING            #
#####################################

print("Saving model...")

uid = uuid.uuid4()

classifier.save_params(f_params='{}/{}-model.pkl'.format(args.output, uid),
                       f_optimizer='{}/{}-opt.pkl'.format(args.output, uid),
                       f_history='{}/{}-history.json'.format(args.output, uid))

logger.experiment.log_text('uid', str(uid))
logger.experiment.log_text('test_name', args.trainlabels)
logger.experiment.log_artifact('{}/{}-model.pkl'.format(args.output, uid))
logger.experiment.log_artifact('{}/{}-opt.pkl'.format(args.output, uid))
logger.experiment.log_artifact('{}/{}-history.json'.format(args.output, uid))
logger.experiment.stop()

print("Saving completed...")
