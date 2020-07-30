# histopathology_cancer_detection
Project for HPI SS2020 for the Deep Learning lecture

## Authors:
* Nicolas Alder (Master Data Engineering, Hasso Plattner Institute)
* Eric Fischer (Master Data Engineering, Hasso Plattner Institute)
* Erik Langenhan (Master IT Systems Engineering, Hasso Plattner Institute)
* Nataniel Müller (Master Digital Health, Hasso Plattner Institute)
* Christian Warmuth (Master Data Engineering, Hasso Plattner Institute)
* Simon Witzke (Master Data Engineering, Hasso Plattner Institute)

## Assignment
Histopathologic cancer detection on hematoxylin and eosin (H\&E)-stained lymph node sections slides following the [Kaggle challenge](https://www.kaggle.com/c/histopathologic-cancer-detection).

## Running and Training on Colab
[How to install and let it run can be viewed in this demo.](https://colab.research.google.com/drive/1ADuwEhJckgJQxXOp42X2fpj0DuDvU3XW?usp=sharing)

Our training colab notebooks can be viewed here:
* [VGGNet11](https://colab.research.google.com/drive/1lfhyK8n9yQuLZ3IC5TGXTAszDZ2MdEAq?usp=sharing)
* [VGGNet19](https://colab.research.google.com/drive/12nWoFbQWahVAjyh1iRCav0hV-EgHjxKB?usp=sharing)
* [ResNet18](https://colab.research.google.com/drive/1haDoIiA51HftiioXyFWYMi_tIj3wIDSY?usp=sharing)
* [ResNet152](https://colab.research.google.com/drive/1UrwvLjxo9StS9NUIj1sGOwO9LurMOal1?usp=sharing)
* [DenseNet121](https://colab.research.google.com/drive/17xaj6wSZunO4TxkJ_E457dwpdwy3X6Qu?usp=sharing)
* [DenseNet201](https://colab.research.google.com/drive/1iCdhbz7fglXh07hDT9IBzGAZD5o30jLp?usp=sharing)

## Setup and Training on Server

**Setup**

``` 
pip3 install -r requirements.txt
```

**Training**

``` 
python3 -W ignore train.py -trnl "path to train split" -tstl "path to test split" -f "path to images" -o "path to output" -m densenet121 -n "Neptune Text"
```

**Testing**

```
python3 test.py -tstl "path to test split" -f "path to images" -a densenet121 -p uuid-model.pkl -o uuid-opt.pkl  -hist uuid-history.json
```

## Experiment Documentation

For documentation and comprehensibility, we stored all our runs in a Neptune.ai project [here](https://ui.neptune.ai/elangenhan/hcd-experiments/experiments?viewId=9ed5b62a-b40b-45fa-b091-81d23be85546). 


![Neptune Experiment Overview](exploration/neptune_overview.jpg?raw=true "Title")

![Neptune Experiment Details](exploration/neputune_graphs.jpg?raw=true "Title")


## Code References

For normalization, we used and modified the [python implementation](https://github.com/schaugf/HEnorm_python) of the normalization method [proposed by Macenko et al.](https://ieeexplore.ieee.org/document/5193250) under the following [licence](https://github.com/schaugf/HEnorm_python/blob/master/licence.txt) (a copy also visible in our repo [here](licence_normalization.txt)). This python implementation is used in the file [helper_scripts/normalization.py](helper_scripts/normalization.py).



