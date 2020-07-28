#!/bin/bash

while true; do
    for name in  'resnet152_96' 'resnet18_96' 'densenet121' 'densenet201' 'vgg11' 'vgg19'   ; do 
        python3 -W ignore server_model_training.py -trnl ./data/train_split_normalized.csv -tstl ./data/test_split_normalized.csv -f ./data/train_normalized/ -o ./data -m $name -n "Refactored Normalized"  ;
    done;
done;

