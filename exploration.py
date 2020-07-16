import pandas
from dplython import (DplyFrame, X, diamonds, select, sift, sample_n, sample_frac, head, arrange, mutate, group_by, summarize, DelayFunction)
from pandas import read_csv

data = read_csv("./../train_labels.csv")
print(data)
