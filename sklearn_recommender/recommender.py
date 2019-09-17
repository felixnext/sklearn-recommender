'''Base Class for a recommender (derived from sklearn estimators).'''


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin
from sklearn.multioutput import MultiOutputClassifier


class BaseRecommender(BaseEstimator, ClassifierMixin):
  '''Defines the basic structure for an Recommender.

  TODO: describe how the recommender interacts with data
  '''
  def __init__(self):
    pass

  def fit(self, X, y, sample_weight=None):
    return self

  def predict(self, X):
    pass

  def predict_proba(self, X):
    pass

  def predict_log_proba(self, X):
    pass

  def score(self, X, y, sample_weight=None):
    pass


class SimilarityRecommender(MultiOutputClassifier):
  '''Recommender for similar items based on the `SimilarityTransformer`.

  The goal of this recommender is to recommend similar items to the items that are already given (item-item recommender).

  Args:
    num_items (int): Number of items that should be estimated (if None output all items)
    sort (str): Defines the function used to aggregate items from the user_item matrix (if provided) (potential values are: 'sum', 'mean', 'count')
    ascending (bool): Defines if values should be sorted ascending or descending
  '''
  def __init__(self, num_items=None, sort=None, ascending=False):
    self.num_items = num_items if num_items is not None else -1
    self.ascending = ascending

    # check sort value
    if sort is not None and sort not in ['sum', 'mean', 'count']:
      raise ValueError("Unkown value for sort: {}".format(sort))
    self.sort = sort

    # init matricies
    self.sim_mat = None
    self.user_item = None
    self.sort_mat = None

  def fit(self, X, y=None, sample_weight=None):
    '''Fits the recommender to the given data.

    Args:
      X (DataFrame): Single DataFrame with similarity matrix or tuple/array of form (user-item matrix, similarity matrix)
    '''
    # store the similarity matrix internally
    if isinstance(X, list) or isinstance(X, tuple):
      self.sim_mat = X[1]
      self.user_item = X[0]

      # generate the sort matrix
      if self.sort is not None:
        self.sort_mat = pd.DataFrame(self.user_item.aggregate(self.sort, axis=1), columns=['ranking'])
    else:
      self.sim_mat = X

    return self

  def _retrieve_preds(self, item):
    '''Retrieve the similar items.'''
    pred = self.sim_mat.loc[item]

    # check for sort matrix
    if self.sort_mat is None:
      pred = pred.sort_values(ascending=self.ascending)
    else:
      pred = pd.DataFrame(pred)
      pred.columns = ['similarity']
      pred = pred.join(self.sort_mat).sort_values(by=['similarity', 'ranking'], ascending=self.ascending)

    # drop initial item and return
    pred = pred.drop(labels=[item])
    return pred

  def predict(self, X):
    '''Predicts the desired number of outputs.

    Returns:
      Array of shape (X.shape[0], num_items)
    '''
    res = []
    for item in X:
      try:
        pred = self._retrieve_preds(item)
        res.append(pred.index[:self.num_items])
      except:
        res.append(np.full(self.num_items, np.nan))
    return np.array(res)

  def predict_proba(self, X):
    '''Predicts the desired number of outputs and returns normalized scores with them.

    '''
    res = []
    for item in X:
      try:
        pred = self._retrieve_preds(item)
        res.append(list(zip(pred.index[:self.num_items], pred.values[:self.num_items])))
      except:
        res.append(np.full(self.num_items, (np.nan, np.nan)))
    return np.array(res)


class CrossSimilarityRecommender(MultiOutputClassifier):
  '''Recommender that uses similarities along one dimension and uses them to recommend items along a different dimension.

  The goal of this recommender is to recommend items through the similarity of users (collaborative-filtering)

  ranking matrix should be a series with items as index

  Args:
    num_items (int): The number of items to recommend
    similarity (Recommender): Recommender that is used to generate the similarity scores between different users (should be a function with no input values)
    filter (func): Function that retrieves a pandas series from the user-item matrix and returns a true/false series, which items the user interacted with
    selection (str): Defines the type of item selection if the possible items exceed the result (options: array, sample, ranking)
    rank_mat (DataFrame): Matrix that contains the ranking for individual items to select for `ranking` option (can be passed by fit function Alternatively)
  '''
  def __init__(self, num_items=5, similarity=SimilarityRecommender, filter=None, selection='array', rank_mat=None):
    self.num_items = num_items
    self.estimator = similarity()

    # define sort function
    if filter is None:
      self.filter = lambda x: x > 0
    else:
      self.filter = filter

    self.selection = selection
    self.rank_mat = rank_mat

  def fit(self, X, y=None, sample_weight=None):
    '''Stores structure of the dataset in the recommender.

    Args:
      X (tuple): Tuple/Array of data items in form (user-item-matrix, user-user-similarity-matrix).
    '''
    # fit the internal estimator
    self.estimator.fit([X[0], X[1]], y, sample_weight)
    self.user_item = X[0]
    # check for ranking matrix
    if not isinstance(X, tuple) and len(X) > 2:
      self.rank_mat = X[2]

    return self

  def predict(self, X):
    '''Predicts list of `num_items` for the given user_ids.'''
    # safty checks
    if self.selection == 'ranking' and self.rank_mat is None:
      raise RuntimeError("No Ranking Matrix provided, but ranking option choosen!")

    # prepare data
    res = []

    # retrieve similar users for each input users
    sim_users = self.estimator.predict(X)
    for x, susers in zip(X, sim_users):
      # retrieve items the user already interacted with
      try:
        seen = self.user_item.loc[x]
        seen = np.array(seen[self.filter(seen)].index)
      except:
        seen = []

      # iterate through all similar users and find relevant items
      user_rec = []
      for user in susers:
        try:
          cur_rec = self.user_item.loc[user]
          cur_rec = np.array(cur_rec[self.filter(cur_rec)].index)
        except:
          continue

        # remove items already seen by current users
        cur_rec = np.setdiff1d(cur_rec, seen)

        # combine items check exit condition
        if len(user_rec) + len(cur_rec) >= self.num_items:
          len_rem = self.num_items - len(user_rec)
          # select relevant items based on option
          if self.selection == 'sample':
            cur_rec = np.random.choice(cur_rec, size=len_rem, replace=False)
          elif self.selection == 'ranking':
            # join items based on their value
            cur_rec = np.array(self.rank_mat.loc[cur_rec].sort_values(ascending=False).index)
            cur_rec = cur_rec[:len_rem]
          else:
            cur_rec = cur_rec[:len_rem]

          # append data to final list
          user_rec = np.append(user_rec, cur_rec)
          break
        else:
          user_rec = np.append(user_rec, cur_rec)

      res.append(np.array(user_rec).astype(self.user_item.columns.dtype))
    return np.array(res)

  def predict_proba(self, X):
    pass
