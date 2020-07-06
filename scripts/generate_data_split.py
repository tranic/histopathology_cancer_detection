import pandas as pd
import numpy as np


def split_data(df, split=(0.8, 0.1, 0.1)):
    """
    Splits data and saves splits into .csv files.

    :param df: the dataframe containing the entire training data
    :param split: triple defining the split ratio (train,test,validate)
    """
    df = df
    n = len(df)
    n_train, n_test, n_validate = np.floor(n * np.array(list(split)))
    print("Splitting into train test validate: ", n_train, n_test, n_validate)

    df_train = df.iloc[0:int(n_train), ]
    df_test = df.iloc[int(n_train):int(n_train + n_test), ]
    df_validate = df.iloc[int(n_train + n_test):len(df) + 1, ]

    df_train.to_csv(path_or_buf="../data/train_split.csv", index=False)
    df_test.to_csv(path_or_buf="../data/test_split.csv", index=False)
    df_validate.to_csv(path_or_buf="../data/validate_split.csv", index=False)


if __name__ == '__main__':
    all_df = pd.read_csv("../data/train_labels.csv")
    split_data(df=all_df, split=(0.8, 0.1, 0.1))
