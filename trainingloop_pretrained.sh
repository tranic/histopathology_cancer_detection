#!/bin/bash

while true; do
    for name in 'densenet121' 'densenet201' 'resnet34' 'resnet152'; do 
        python3 server_model_training_pretrained.py -trnl ./data/train_split.csv -tstl ./data/test_split.csv -f ./data/train -o ./data -m $name;
    done;
done;
