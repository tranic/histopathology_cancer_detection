#!/bin/bash

while true; do
    for name in  'resnet152_96' 'resnet18_96' 'densenet121' 'densenet201' 'vgg11' 'vgg19'   ; do 
        python3 -W ignore server_model_training.py -trnl ./data/train_labels.csv -f ./data/train -o ./data -n "Refactored"  ;
    done;
done;
