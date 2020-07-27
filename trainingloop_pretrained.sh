#!/bin/bash

while true; do
    for name in 'resnet34' 'densenet121' 'densenet201'  'resnet152'; do 
        python3 -W ignore server_model_training_pretrained.py -trnl ./data/train_labels.csv -f ./data/train -o ./data -m $name -n "Refactored Pretrained" ;
    done;
done;
