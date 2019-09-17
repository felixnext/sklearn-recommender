# Sklearn Recommender

This library implements some common recommender functions.

## Getting Started

Install the library on your local distribution through:

`pip install .`

(pypi to follow).

## Tutorial

All functions of the library are build around `Transformer` and `Estimator`, allowing them to be used through a `Pipeline` as well as with `GridSearchCV`.
In general the system assumes the input to be

```python
import sklearn_recommender as skr
```

### Transformer

**User-Item**

Uses a list of user-item interactions to create a user-item matrix that can also be used as input to the similarity transformer. This also supports binary interactions by setting the `binarize` flag in the constructur.

```python
tf = skr.transformer.UserItemTransformer(user_col='user_id', item_col='item_id', value_col='ranking', agg_fct='mean')
user_item = tf.transform(df_reviews)
```

**Similarity**

Creates a similarity matrix based on the given input data. It assumes that each row in a matrix is single item and computes the similarity between them according to the features listed in the given `cols`. The resulting dataframe will have the `index_col` as index (or preserves the original index if `index_col` is not given).
Note that `cols` can be a list of names or a tuple defining the position of the first and last column to use.

```python
tf = skr.transformer.SimilarityTransformer(cols=(2, -1), index_col='item_id', normalize=True)
sim_mat = tf.transform(items)
```

**GloVe**

Based on the [Global Vector for Word Embeddings](https://nlp.stanford.edu/projects/glove/), this implements a transform that create n-dimensional word embeddings based on a list of input texts.
The library comes with functions to download pre-trained models from the project website (note of caution: these models can take 3+GB of additional disk space). There are current two pre-trained models integrated: `'wikipedia'` (which only has 300-dimensional embeddings) and `'twitter'` (coming with 25, 50, 100 and 200 dimensional embeddings).
There are also multiple ways to create embeddings for the given text (as it spans more than one word):

* `word` - generates a list of simple word embeddings (only recommended for single words)
* `sent` - creates a document embedding by adding up all vectors and normalizing them
* `sent-matrix` - creates a matrix with `max_feat` rows that contains the embeddings for the first `max_feat` words (if less words it is filled with random vectors according to distribution of vector space)
* `centroid` - Takes all word embeddings in the given text and computes the `max_feat` centroids for the clusters of the vectors

```python
# optionally download the requried models
skr.glove.download('twitter')
tf = skr.glove.GloVeTransformer('twitter', 25, 'sent', tokenizer=skr.nlp.tokenize_clean)
```

### Recommender

**Similarity**

Recommendations are made based on the similarity of item. That requires the id of an item to be given and returns the n most similar candidates.

```python
rec = skr.recommender.SimilarityRecommender(5)
rec.fit(sim_mat)
# predict the 5 most similar items to the given items 5, 6 and 7 respectively
rec.predict([5, 6, 7])
```

**Cross-Similarity**

Collaborative-Filtering based approach. This uses the most similar items on one dimensions (e.g. most similar users to the given user) to predict the most relevant items along a different dimension (e.g. the items the most similar users interacted the most with).

```python
rec = skr.recommender.CrossSimilarityRecommender(5)
rec.fit((user_item, sim_mat, ))
rec.predict([10, 12])
```

### Helper Functions

Apart from the sklearn extensions, there are also various

**Train-Test Split:**

Train-Test split for

```python
df = ...
# create a 30% size test set
train_df, test_df = skr.train_test_split(df, split_size=0.3)
```

**NLP Functions:**

In combination with text embeddings, there are some functions to tokenize input words using functions from `nltk`.

## Design Philosophy

TODO

## Future Work

* Implement sufficient test coverage
* Add type tests to code (+ conversions to required datatypes between numpy and pandas)
* Implement additional guarantees into the `train_test_split` (e.g. coverage of item ids)

## License

The code is published under MIT License
