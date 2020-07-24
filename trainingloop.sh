#!/bin/bash

while true; do
    for name in "densnet121" "ResNet" "asdasdasd"; do 
        python3 server_model_training.py -trnl ./data/train_split.csv -tstl ./data/test_split.csv -f ./data/train -o ./data -m $name;
    done;
done;
