#!/bin/bash

while true; do
    for name in 'resnet18_96' 'vgg19' 'vgg11' 'resnet152_96'  'densenet121' 'densenet201'     ; do 
        python3 -W ignore server_model_training.py -trnl ./data/train_small.csv -f ./data/normalized_data/ -o ./data -m $name;
    done;
done;
