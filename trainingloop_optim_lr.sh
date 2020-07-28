#!/bin/bash

while true; do
    for name in  'vgg11' 'vgg19'   ; do 
        python3 -W ignore server_model_training_optim_lr_change.py -trnl ./data/train_labels.csv -f ./data/train -o ./data -m $name -n "Refactored Adam"  ;
    done;
done;
