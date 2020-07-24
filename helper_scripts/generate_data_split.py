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
    print("Number of images found: ", n)

    if len(split) == 3: # train, test, validation split
        n_train, n_test, n_validate = np.floor(n * np.array(list(split)))
        print("Splitting into train test validate: ", n_train, n_test, n_validate)

        df_train = df.iloc[0:int(n_train), ]
        df_test = df.iloc[int(n_train):int(n_train + n_test), ]
        df_validate = df.iloc[int(n_train + n_test):len(df) + 1, ]

        # show label distributions
        print(df_train["label"].value_counts())
        print("Prop. positive train=", df_train["label"].value_counts()[1]/df_train["label"].value_counts()[0])
        print(df_test["label"].value_counts())
        print("Prop. positives test=", df_test["label"].value_counts()[1]/df_test["label"].value_counts()[0])
        print(df_validate["label"].value_counts())
        print("Prop. positives validate=", df_validate["label"].value_counts()[1] / df_validate["label"].value_counts()[0])

        # save splits
        df_train.to_csv(path_or_buf="../data/train_split.csv", index=False)
        df_test.to_csv(path_or_buf="../data/test_split.csv", index=False)
        df_validate.to_csv(path_or_buf="../data/validate_split.csv", index=False)
    elif len(split) == 2: # train, test split
        n_train, n_test = np.floor(n * np.array(list(split)))
        print("Splitting into train test: ", n_train, n_test)

        df_train = df.iloc[0:int(n_train), ]
        df_test = df.iloc[int(n_train):len(df) + 1, ]

        # show label distributions
        print(df_train["label"].value_counts())
        print("Prop. positive train=", df_train["label"].value_counts()[1] / df_train["label"].value_counts()[0])
        print(df_test["label"].value_counts())
        print("Prop. positives test=", df_test["label"].value_counts()[1] / df_test["label"].value_counts()[0])

        # save splits
        df_train.to_csv(path_or_buf="../data/train_split.csv", index=False)
        df_test.to_csv(path_or_buf="../data/test_split.csv", index=False)
    else:
        raise Exception("Error: split has to be tuple of length 2 or 3.")


if __name__ == '__main__':
    all_df = pd.read_csv("../data/train_labels.csv")
    split_data(df=all_df, split=(0.8, 0.2)) # for train, test only
    # split_data(df=all_df, split=(0.8, 0.2, 0))
