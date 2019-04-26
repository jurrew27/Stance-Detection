from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords

# TODO X to lowercase

class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec=None):
        self.word2vec = word2vec
        self.dim = word2vec.vector_size

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = word2vec.vector_size

    def fit(self, x, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(x)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


# TODO strip handles
# TODO split hashtags on upper case, see https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
class SentenceSplitter(BaseEstimator):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        token = simple_preprocess(X.replace('\'', ''), max_len=30)
        return list(filter(lambda x: x not in self.stop_words, token))
