# histopathology_cancer_detection
Project for HPI SS2020 for the seminar Deep Learning 

## Authors:
* Nicolas Alder (Master Data Engineering, Hasso Plattner Institute)
* Eric Fischer (Master Data Engineering, Hasso Plattner Institute)
* Erik Langenhan (Master IT Systems Engineering, Hasso Plattner Institute)
* Nataniel Müller (Master Digital Health, Hasso Plattner Institute)
* Christian Warmuth (Master Data Engineering, Hasso Plattner Institute)
* Simon Witzke (Master Data Engineering, Hasso Plattner Institute)

## Assignment
Histopathologic cancer detection on hematoxylin and eosin (H\&E)-stained lymph node sections slides following the [Kaggle challenge](https://www.kaggle.com/c/histopathologic-cancer-detection).

## Installation & Running
Where data needs to be located

How to run

## Results

Picture Einfügen

## Experiment Documentation

For documentation and comprehensibility, we stored all our runs in a Neptune.ai project [here](https://ui.neptune.ai/elangenhan/hcd-experiments/experiments?viewId=9ed5b62a-b40b-45fa-b091-81d23be85546). 


![Neptune Experiment Overview](exploration/neptune_overview.jpg?raw=true "Title")

![Neptune Experiment Details](exploration/neputune_graphs.jpg?raw=true "Title")


## Code we used

For normalization, we used and modified the [python implementation](https://github.com/schaugf/HEnorm_python) of the normalization method [proposed by Macenko et al.](https://ieeexplore.ieee.org/document/5193250) under the following [licence](https://github.com/schaugf/HEnorm_python/blob/master/licence.txt) (a copy also visible in our repo [here](licence_normalization.txt)). This python implementation is used in the file [/helper_scripts/normalization.py]([/helper_scripts/normalization.py]).



