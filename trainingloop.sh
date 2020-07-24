#!/bin/bash

while true; do
    for name in 'vgg11' 'vgg19' 'densenet121' 'densenet201' 'resnet18' 'resnet152' 'resnet18_96' 'resnet152_96'; do 
        python3 server_model_training.py -trnl ./data/train_split.csv -tstl ./data/test_split.csv -f ./data/train -o ./data -m $name;
    done;
done;
