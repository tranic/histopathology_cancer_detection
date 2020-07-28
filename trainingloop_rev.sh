#!/bin/bash

while true; do
    for name in  'vgg19' 'vgg11' 'resnet18_96' 'resnet152_96' 'densenet201' 'densenet121' ; do 
        python3 -W ignore server_model_training.py -trnl ./data/train_split.csv -tstl ./data/test_split.csv -f ./data/train -o ./data -m $name -n "Refactored"  ;
    done;
done;
