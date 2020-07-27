#!/bin/bash

while true; do
    for name in  'vgg11' 'vgg19''resnet18_96'  'densenet121' 'densenet201' 'resnet152_96'    ; do 
        python3 -W ignore server_model_training_epoch_100.py -trnl ./data/train_labels.csv -f ./data/train -o ./data -m $name -n "Refactored"  ;
    done;
done;
