'''Contains additional helper functions.'''


import pandas as pd
import numpy as np


def train_test_split(X, split_size=0.2, random_state=None):
  '''Performs a split of the data into a test and train set.

  The data is to be expected as single datapoints (e.g. user | item | value)

  Args:
    X (pd.DataFrame): Dataframe that contains the data points
    split_size (float): Percent of data used for testing
    random_state (int): init value used for random splitting (to create deterministic behaviour)

  Returns:
    train: Dataframe containing training datapoints
    test: Dataframe containing testing datapoints
  '''
  # TODO: implement functions to ensure all items are present in both datasets
  # random permuation of input
  rand = X.sample(frac=1, random_state=random_state)
  n_items = int(X.shape[0] * split_size)
  # split the dataset
  test = X.iloc[:n_items]
  train = X.iloc[n_items:]

  return train, test
