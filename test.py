import os
import argparse
import pandas as pd

from sklearn import metrics

from skorch import NeuralNetBinaryClassifier

from torchvision import transforms

from data_loading import HistopathDataset
from architecture import VGG11, VGG19, DenseNet121, DenseNet201, ResNet18_96, ResNet152_96, LeNet


######################################
#        COMMAND LINE PARAMS        #
#####################################

parser = argparse.ArgumentParser(description='Set necessary values to train different types of predefined models')
parser.add_argument("--testlabels", "-tstl", help="set test label path")
parser.add_argument("--files", "-f", help="set file  path")

parser.add_argument("--architecture", "-a", help="set model param path")

parser.add_argument("--parameter", "-p", help="set model param path")
parser.add_argument("--optimizer", "-o", help="set model optimizer path")
parser.add_argument("--history", "-hist", help="set model history path")

args = parser.parse_args()


######################################
#            DATA LOADING           #
#####################################

dataset_test = HistopathDataset(
        label_file = os.path.abspath(args.testlabels),
        root_dir = os.path.abspath(args.files),
        transform = transforms.ToTensor(),
        in_memory = False)


df = pd.read_csv(args.testlabels)
target = df["label"]    


######################################
#      LOAD TEST ARCHITECTURE       #
#####################################

arch_switcher = {'vgg11': VGG11,
                  'vgg19': VGG19,
                  'densenet121': DenseNet121,
                  'densenet201': DenseNet201,
                  'resnet18_96': ResNet18_96,
                  'resnet152_96': ResNet152_96,
                  'lenet': LeNet}

get_arch = arch_switcher.get(args.architecture, lambda: "Architecture does not exist")


net = NeuralNetBinaryClassifier(
      module=get_arch())
  
net.initialize()
net.load_params(f_params=args.parameter, 
                f_optimizer=args.optimizer, 
                f_history=args.history)



######################################
#        CALCULATE METRICS          #
#####################################

print("Predicting lables...")
y_hat = net.predict(dataset_test)
print("Calculating accuracy...")
accuracy = metrics.accuracy_score(target, y_hat)
print("Calculating precision...")
precision = metrics.precision_score(target, y_hat)
print("Calculating recall...")
recall = metrics.recall_score(target, y_hat)
print("Calculating F_1...")
f1 = metrics.f1_score(target, y_hat)

print("Predicting probabilities...")
y_hat = net.predict_proba(dataset_test)
print("Calculating AUROC...")
roc_auc = metrics.roc_auc_score(target, y_hat[:, 1])



print('''Meticts for the supplied architecture {} 
              \033[1mAccuracy:\033[0m  {}
              \033[1mPrecision:\033[0m {}
              \033[1mRecall:\033[0m    {}
              \033[1mF_1:\033[0m       {}
              \033[1mAUROC:\033[0m     {}
              '''
          .format(net.module,
                  accuracy,
                  precision,
                  recall,
                  f1,
                  roc_auc))

