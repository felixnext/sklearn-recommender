'''
Various helper function (especially in the pre-processing of the data)
'''

import time
# pre-processing
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.chunk import tree2conlltags
import re
# metrics
from sklearn.metrics import f1_score, precision_score, recall_score

# make sure relevant nltk data is loaded
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('omw')

# init these globally (to avoid call for every run)
stops = set(stopwords.words('english'))
wlem = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    '''Matches the given treebank POS tag to a wordnet one to be used by lemmatizer.'''
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return 'n'

def tokenize(text):
    '''Tokenizes the words in the text, uses lemmatization and POS tagging.'''
    # remove punctuation
    text = re.sub("[\.,\\:;!?'\"-]", " ", text.lower())
    tokens = word_tokenize(text)

    # pos tags and remove stopwords
    tags = pos_tag(tokens)
    tags = filter(lambda x: x[0] not in stops, tags)

    # part of speech
    tags = map(lambda x: (wlem.lemmatize(x[0], pos=get_wordnet_pos(x[1])), x[1]), tags)

    return list(tags)

def tokenize_clean(text):
    '''Removes POS Tags from the words.'''
    return list(map(lambda x: x[0], tokenize(text)))

def tokenize_ner(text):
    '''Applies Named Entity Recognition to the words.'''
    # remove punctuation
    text = re.sub("[\.,\\:;!?'\"-]", " ", text.lower())
    sents = sent_tokenize(text)

    tokens = chain.from_iterable(map(lambda x: tree2conlltags( ne_chunk(pos_tag(word_tokenize(x))) ), sents))
    tokens = filter(lambda x: x[0] not in stops, tokens)
    tokens = map(lambda x: (wlem.lemmatize(x[0], pos=get_wordnet_pos(x[1])), x[1], x[2]), tokens)

    return list(tokens)

def extract_ner(text):
    '''Extracts a list of Named Entities as additional feature.'''
    text = re.sub("[\.,\\:;!?'\"-]", " ", text.lower())
    sents = sent_tokenize(text)
    tokens = chain.from_iterable(map(lambda x: tree2conlltags( ne_chunk(pos_tag(word_tokenize(x))) ), sents))
    tokens = filter(lambda x: x[2] != "O", tokens)

    return list(tokens)

def _metric(y_test, y_pred, avg='binary'):
    '''Returns scores of the model.'''
    f1 = f1_score(y_test, y_pred, average=avg)
    prec = precision_score(y_test, y_pred, average=avg)
    rec = recall_score(y_test, y_pred, average=avg)
    return f1, prec, rec

def score_and_doc(model, name, X_test, y_test, extended=False):
    '''General scoring function.

    Takes the model and predicts output against given data.
    Finally scores them along different metrics and writes the results to the experiments file.

    Args:
        model: The sklearn model
        name (str): name of the model (used for documentation)
        X_test: test data
        y_test: expected test labels
    '''
    # predict the data
    y_pred = model.predict(X_test)

    # score the model (general)
    avg = 'micro'
    f1, prec, rec = _metric(y_test, y_pred, avg)

    # retrieve current config data?
    config = str(model.get_params())
    config = re.sub("[\n\t]", " ", config)
    config = re.sub("[,]", "/", config)
    config = re.sub("[ ]+", " ", config)

    # open file and append
    f=open("../experiments.csv", "a+")
    f.write("{},{:.6f},{:.6f},{:.6f},{},{}\n".format(name, f1, prec, rec, time.ctime(), config))
    f.close()

    # print output
    print("{}: F1 Score = {:.6f} (P={:.6f} / R={:.6f})".format(name, f1, prec, rec))

    # score the model (class-wise)
    if extended:
        # calculate the score for each cateogry
        cats = y_test.columns
        for i, c in enumerate(cats):
            sf1, sprec, srec = _metric(y_test.iloc[:, i], y_pred[:, i])
            print("  {:25} F1 Score = {:.6f} (P={:.6f} / R={:.6f})".format(c + ":", sf1, sprec, srec))
