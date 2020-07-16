#!/bin/bash

DATA="https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/11848/862157/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1595090618&Signature=GfcYZgp3ciQZL%2B05cSpxD65ktQEl%2B1R5e1LTmk0jp9zYqL38gUtrFS83md8WEkIxD%2Fa3P%2F6rUuHStotD3Yv8LSQ%2Fizuv2h4kK3TTnN0h7LkPiLvGyc6hQHmND0UraVGe03sAdrfQWu8HDLS%2BZKENzY9Ca30z8xR5XEMYpk%2Be0KboGQ%2FwMMhKbdv9S09cNOheKLfiH7JIYjYAy3rIFuIPhHezRAEdWVRvnLHM4bmScVRRIJdnNYxptmYCdX4ztwlY75A%2FDIUW%2Bqk8EKlwSl1HVwAx3gP14oid%2F34frBTffdwskudA6%2FxNdzbSqeTDmqQWprnOels6VSG6hOKC5zcHTQ%3D%3D&response-content-disposition=attachment%3B+filename%3Dhistopathologic-cancer-detection.zip"
DATA_TARGET_PATH="/content/drive/My Drive/cancerprediction"

if [ ! -d "/content/drive/" ]
then
  echo "Google Drive not mounted - aborting"
  exit 1
fi

if [ ! -d "$DATA_TARGET_PATH" ]
then
    wget -x $DATA -O data.zip
    unzip data.zip -d "$DATA_TARGET_PATH"
fi

TRAIN_LABELS="./train_labels.csv"

while IFS="" read -r p || [ -n "$p" ]
do
    FOLDER=$(echo ${p} | awk -F'/' '{print $2}')
    FILE=$(echo ${p} | awk -F'/' '{print $3}' | awk -F',' '{print $1}')
    FILE="$FILE.tif"
    mkdir -p $FOLDER
    mv $FILE $FOLDER/
done < $TRAIN_LABELS
