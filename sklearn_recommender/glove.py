'''Module to load and use GloVe Models.

Code Inspiration from:
https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout
'''

import os
import numpy as np
import pandas as pd
import urllib.request
from zipfile import ZipFile
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

# TODO: store in user home dir
folder = os.path.dirname(os.path.realpath(__file__))

def download(name):
  '''Downloads the relevant dataset and extracts it.

  Args:
    name (str): Name of the model to download (options are: [twitter, wikipedia])

  Returns:
    True if successful, otherwise False
  '''
  # check if files exists
  if os.path.isfile(os.path.join(folder, '{}.zip'.format(name))):
    print('File found, no download needed')
    return True

  url = None
  if name == 'twitter':
    url = 'http://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip'
  elif name == 'wikipedia':
    url = 'http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip'
  if url is not None:
    try:
      urllib.request.urlretrieve(url, os.path.join(folder, '{}.zip'.format(name)))
    except:
      print("download failed")
      return False
    try:
      # Create a ZipFile Object and load sample.zip in it
      with ZipFile(os.path.join(folder, '{}.zip'.format(name)), 'r') as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall(folder)
      return True
    except:
      print("extraction failed")
      return False
  return False

class GloveEmbeddings:
  '''Class to load embeddings model and generate it for words or sentences.'''
  def __init__(self, name, dim=25):
    # load data
    self.emb = self.load_vectors(name, dim)
    self.emb_size = dim
    # calculate items for randomization (explicit convert to list to avoid numpy warning)
    all_embs = np.stack(list(self.emb.values()))
    self.emb_mean,self.emb_std = all_embs.mean(), all_embs.std()

  def get_coefs(self, word, *arr):
    '''Helper Function to transform the given vector into a float array.'''
    return word, np.asarray(arr, dtype='float32')

  def load_vectors(self, name, dim):
    '''Load the given vector data.'''
    # retrieve file name
    file = None
    if name == 'twitter':
      file = os.path.join(folder, 'glove.{}.27B.{}d.txt'.format(name, dim))
    elif name == 'wikipedia':
      file = os.path.join(folder, 'glove.840B.{}d.txt'.format(dim))
    else:
      raise ValueError('Unkown model type ({})'.format(name))
    # load the embeddings
    with open(file, encoding='utf-8') as file:
      embeddings_index = [self.get_coefs(*o.strip().split()) for o in file]
    embeddings_index = list(filter(lambda x: len(x[1]) == dim, embeddings_index))
    return dict(embeddings_index)

  def word_vector(self, word):
    '''Tries to retrieve the embedding for the given word, otherwise returns random vector.'''
    # generate randomness otherwise
    vec = self.emb.get(word)
    return vec if vec is not None else np.random.normal(self.emb_mean, self.emb_std, (self.emb_size))

  def sent_vector(self, sent, use_rand=True):
    '''Generates a single embedding vector.

    Args:
      sent (list): List of tokenized words to use
      use_rand (bool): Defines if unkown words should be filled with random vectors (otherwise only use known vectors)

    Returns:
      Single normalized Vector to be used as embedding
    '''
    vec = None
    vec_count = 0
    for word in sent:
      wvec = self.emb.get(word)
      if wvec is None and use_rand:
        wvec = np.random.normal(self.emb_mean, self.emb_std, (self.emb_size))
      if wvec is not None:
        if vec is None:
          vec = wvec
        else:
          vec += wvec
      vec_count += 1
    # normalize the vector
    if vec is not None and vec_count > 0:
      vec = vec / vec_count
    # if no word is found return random vector
    return vec if vec is not None else np.random.normal(self.emb_mean, self.emb_std, (self.emb_size))

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
    for i, word in enumerate(sent):
      if i >= max_feat: continue
      vec = self.emb.get(word)
      if vec is not None: embedding_matrix[i] = vec
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
      if vec is not None: vecs.append(vec)

    # return random vector if none found
    if len(vecs) < max_feat:
      return np.array(vecs + [np.random.normal(self.emb_mean, self.emb_std, (self.emb_size)) for i in range(max_feat - len(vecs))])
    elif len(vecs) == max_feat:
      return np.array(vecs)

    # perform clustering
    kmeans = KMeans(n_clusters=max_feat).fit(vecs)

    # return the centroid vectors
    return kmeans.cluster_centers_

class GloVeTransformer(BaseEstimator, TransformerMixin):
  '''Transformer for the GloVe Model.'''

  def __init__(self, name, dim, type, tokenizer, max_feat=None):
    '''Create the Transformer.

    Note that the centroid option might be slow.

    Args:
      name (str): Name of the model
      dim (int): Number of dimensions to use
      type (str): Type of the transformation (options are: ['word', 'sent', 'sent-matrix', 'centroid'])
      tokenizer (fct): Function to tokenize the input data
      max_feat (int): Number of maximal feature vectors used per input
    '''
    # safty checks
    if type not in ['word', 'sent', 'sent-matrix', 'centroid']:
      raise ValueError("Invalid value for type: ({})".format(type))
    if type in ['sent-matrix', 'centroid'] and max_feat is None:
      raise ValueError("Required value for max_feat for type ({})".format(type))
    # set values
    self.glove = GloveEmbeddings(name, dim)
    self.type = type
    self.tokenizer = tokenizer
    self.max_feat = max_feat

  def fit(self, x, y=None):
    return self

  def vectors(self, text):
    '''Extracts the specified type of vector for the given input data.'''
    # retrieve the vectors
    tokens = self.tokenizer(text)
    if self.type == 'word':
      return np.concat([self.glove.word_vector(tok) for tok in tokens])
    elif self.type == 'sent':
      return self.glove.sent_vector(tokens)
    elif self.type == 'sent-matrix':
      # note: use padding to avoid pipeline problems
      return self.glove.sent_matrix(tokens, self.max_feat, True).reshape([-1])
    elif self.type == 'centroid':
      return self.glove.centroid_vectors(tokens, self.max_feat).reshape([-1])
    return np.nan

  def transform(self, X):
    X_tagged = pd.Series(X).apply(lambda x: pd.Series(self.vectors(x)))
    df = pd.DataFrame(X_tagged).fillna(0).replace([-np.inf], -1).replace([np.inf], 1)
    return df
