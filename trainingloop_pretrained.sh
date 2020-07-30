#!/bin/bash

while true; do
    for name in 'resnet152' 'resnet34' 'densenet121' 'densenet201'  ; do 
        python3 -W ignore train_pretrained.py -trnl ./data/train_labels.csv -f ./data/train -o ./data -m $name -n "Refactored Pretrained" ;
    done;
done;
