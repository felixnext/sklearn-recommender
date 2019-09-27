'''Module to load and use GloVe Models.

Code Inspiration from:
https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout
'''

import os, warnings
import numpy as np
import pandas as pd
import urllib.request
from zipfile import ZipFile
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def file_path(cache_dir=None, create=False):
  '''Generates the full file-path based on the given name.'''
  # update the directory based on the current path
  if cache_dir is None:
    cache_dir = os.path.join(os.path.expanduser('~'), '.sklearn-recommender')
  cache_path = os.path.expanduser(cache_dir)
  # attempt to create directory
  if create and not os.path.exists(cache_path):
    os.makedirs(cache_path)
  # check additional access rights
  if not os.access(cache_path, os.W_OK):
    warnings.warn("No access to default data dir: {} - using /tmp instead".format(cache_path))
    cache_path = os.path.join('/tmp', '.sklearn-recommender')

  return cache_path

def download(name, cache_dir=None):
  '''Downloads the relevant dataset and extracts it.

  Args:
    name (str): Name of the model to download (options are: [twitter, wikipedia])

  Returns:
    True if successful, otherwise False
  '''
  folder = file_path(cache_dir, create=True)
  file = os.path.join(folder, '{}.zip'.format(name))
  # check if files exists
  if os.path.isfile(file):
    print('File found, no download needed')
    return True

  url = None
  if name == 'twitter':
    url = 'http://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip'
  elif name == 'wikipedia':
    url = 'http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip'
  if url is not None:
    try:
      urllib.request.urlretrieve(url, file)
    except:
      warnings.warn("download of {} data failed".format(name))
      return False
    try:
      # Create a ZipFile Object and load sample.zip in it
      with ZipFile(file, 'r') as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall(folder)
      return True
    except:
      warnings.warn("extraction of {} data failed".format(name))
      return False
  return False

class GloveEmbeddings:
  '''Class to load embeddings model and generate it for words or sentences.'''
  def __init__(self, name, dim=25, cache_dir=None):
    # load data
    self.emb = self.load_vectors(name, dim, cache_dir)
    self.emb_size = dim
    # calculate items for randomization (explicit convert to list to avoid numpy warning)
    all_embs = np.stack(list(self.emb.values()))
    self.emb_mean,self.emb_std = all_embs.mean(), all_embs.std()

  def get_coefs(self, word, *arr):
    '''Helper Function to transform the given vector into a float array.'''
    return word, np.asarray(arr, dtype='float32')

  def load_vectors(self, name, dim, cache_dir=None):
    '''Load the given vector data.'''
    # retrieve file name
    file = None
    if name == 'twitter':
      file = 'glove.{}.27B.{}d.txt'.format(name, dim)
    elif name == 'wikipedia':
      file = 'glove.840B.{}d.txt'.format(dim)
    else:
      raise ValueError('Unkown model type ({})'.format(name))
    file = os.path.join(file_path(cache_dir), file)
    # check if file exists
    if not os.path.exists(file):
      raise IOError("The given dimension is not available or the model is not downloaded ({})!".format(file))
    # load the embeddings
    with open(file, encoding='utf-8') as file:
      embeddings_index = [self.get_coefs(*o.strip().split()) for o in file]
    embeddings_index = list(filter(lambda x: len(x[1]) == dim, embeddings_index))
    return dict(embeddings_index)

  def word_vector(self, word, normalize=True):
    '''Tries to retrieve the embedding for the given word, otherwise returns random vector.'''
    # generate randomness otherwise
    vec = self.emb.get(word)
    if vec is None:
      vec = np.random.normal(self.emb_mean, self.emb_std, (self.emb_size))
    else:
      vec = np.copy(vec)
    # check for normalization
    if normalize:
      norm = np.linalg.norm(vec)
      if norm != 0:
        vec = np.divide(vec, norm)
    return vec

  def sent_vector(self, sent, use_rand=True):
    '''Generates a single embedding vector.

    Args:
      sent (list): List of tokenized words to use
      use_rand (bool): Defines if unkown words should be filled with random vectors (otherwise only use known vectors)

    Returns:
      Single normalized Vector to be used as embedding
    '''
    vec = None
    for word in sent:
      wvec = self.emb.get(word)
      if wvec is None and use_rand:
        wvec = np.random.normal(self.emb_mean, self.emb_std, (self.emb_size))
      if wvec is not None and len(wvec) > 0:
        wvec = np.copy(wvec)
        if vec is None:
          vec = wvec
        else:
          vec += wvec

    # select the vector
    if vec is None:
      warnings.warn('No word vector was found for combination: {} - using random vector (to silence this warning activate `use_rand`)'.format(str(sent)))
      vec = np.random.normal(self.emb_mean, self.emb_std, (self.emb_size))
    # normalize the vector
    norm = np.linalg.norm(vec)
    if norm != 0:
      vec = np.divide(vec, norm)
    # if no word is found return random vector
    return vec

  def sent_matrix(self, sent, max_feat, pad, dedub=False):
    '''Generates a Matrix of single embeddings for the item.

    Args:
      sent (list): List of tokenized words
      max_feat (int): Number of maximal features to extract
      pad (bool): Defines if the resulting matrix should be zero-padded to max_feat
      dedub (bool): Defines if the word list should be de-duplicated

    Returns:
      2-D Matrix with dimensions [max_feat, embedding_size]
    '''
    # remove duplicates
    if dedub:
      sent = list(set(sent))
    # setup matrix
    nb_words = min(max_feat, len(sent))
    embedding_matrix = np.random.normal(self.emb_mean, self.emb_std, (nb_words, self.emb_size))
    # iterate through all words
    i = 0
    for word in sent:
      if i >= max_feat: continue
      vec = self.emb.get(word)
      if vec is not None:
        embedding_matrix[i] = np.copy(vec)
        i += 1
    # pad the matrix to max features
    if pad and nb_words < max_feat:
      embedding_matrix = np.pad(embedding_matrix, (max_feat, self.emb_size), 'constant', constant_values=[0])
    return embedding_matrix

  def centroid_vectors(self, sent, max_feat):
    '''Generates a list of `max_feat` vectors to be used as representation.

    Args:
      sent (list): Tokenized words in the document
      max_feat (int): Number of vectors to generate

    Returns:
      Array of centroid vectors for the given document
    '''
    # generate list of vectors (use set as order not relevant and to avoid duplicates)
    vecs = []
    for word in set(sent):
      vec = self.emb.get(word)
      if vec is not None: vecs.append(np.copy(vec))

    # return random vector if none found
    if len(vecs) < max_feat:
      warnings.warn('Less than expected clusters found ({} to {}) - filling with random vectors'.format(len(vecs), max_feat))
      return np.array(vecs + [np.random.normal(self.emb_mean, self.emb_std, (self.emb_size)) for i in range(max_feat - len(vecs))])
    elif len(vecs) == max_feat:
      return np.array(vecs)

    # perform clustering
    kmeans = KMeans(n_clusters=max_feat).fit(vecs)

    # return the centroid vectors
    return kmeans.cluster_centers_

class GloVeTransformer(BaseEstimator, TransformerMixin):
  '''Transformer for the GloVe Model.'''

  def __init__(self, name, dim, type, tokenizer, max_feat=None, use_random=False, cache_dir=None):
    '''Create the Transformer.

    Note that the centroid option might be slow.

    Args:
      name (str): Name of the model
      dim (int): Number of dimensions to use
      type (str): Type of the transformation (options are: ['word', 'sent', 'sent-matrix', 'centroid'])
      tokenizer (fct): Function to tokenize the input data
      max_feat (int): Number of maximal feature vectors used per input
      use_random (bool): Defines if random fill should be used for unkown words
      cache_dir (str): Directory to store embeddings (if previously donwloaded) (default: ~./sklearn-recommender)
    '''
    # safty checks
    if type not in ['word', 'sent', 'sent-matrix', 'centroid']:
      raise ValueError("Invalid value for type: ({})".format(type))
    if type in ['sent-matrix', 'centroid'] and max_feat is None:
      raise ValueError("Required value for max_feat for type ({})".format(type))
    # set values
    self.glove = GloveEmbeddings(name, dim, cache_dir=cache_dir)
    self.type = type
    self.tokenizer = tokenizer
    self.max_feat = max_feat
    self.use_rand = use_random

  def fit(self, x, y=None):
    return self

  def vectors(self, text):
    '''Extracts the specified type of vector for the given input data.'''
    # retrieve the vectors
    tokens = self.tokenizer(text)
    if self.type == 'word':
      return np.concat([self.glove.word_vector(tok) for tok in tokens])
    elif self.type == 'sent':
      return self.glove.sent_vector(tokens, self.use_rand)
    elif self.type == 'sent-matrix':
      # note: use padding to avoid pipeline problems
      return self.glove.sent_matrix(tokens, self.max_feat, True).reshape([-1])
    elif self.type == 'centroid':
      return self.glove.centroid_vectors(tokens, self.max_feat).reshape([-1])
    warnings.warn('Unkown type {} - `NaN` returned'.format(self.type))
    return np.nan

  def transform(self, X):
    X_tagged = pd.Series(X).apply(lambda x: pd.Series(self.vectors(x)))
    df = pd.DataFrame(X_tagged).fillna(0).replace([-np.inf], -1).replace([np.inf], 1)
    return df
