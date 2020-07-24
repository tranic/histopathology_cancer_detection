#!/bin/bash
TRAIN_LABELS="../data/train_labels.csv"
FILES_PER_SUBFOLDER=20000

i=0
while IFS="" read -r p || [ -n "$p" ]
do
    i=$((i+1))
    echo "./$((i/FILES_PER_SUBFOLDER))/${p}" >> train_labels.csv
done < $TRAIN_LABELS