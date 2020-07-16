#!/bin/bash

TRAIN_LABELS="./train_labels.csv"
FOLDER="./data_splitted/1/"
FILES_PER_SUBFOLDER=20000

while IFS="" read -r p || [ -n "$p" ]
do
	FILE="$(echo $p | awk -F',' '{print $1}')"
	FILE="$FILE.tif"
	FILE="$FOLDER$FILE"

    	if test -f "$FILE"; then
		echo "${p}" >> train_labels_batch.csv
    	fi
done < $TRAIN_LABELS
