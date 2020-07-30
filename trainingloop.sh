#!/bin/bash

while true; do
    for name in  'densenet121' 'densenet201' 'resnet152_96' 'resnet18_96' 'vgg11' 'vgg19'   ; do 
        python3 -W ignore train.py -trnl ./data/train_split.csv -tstl ./data/test_split.csv -f ./data/train -o ./data -m $name -n "Refactored"  ;
    done;
done;
