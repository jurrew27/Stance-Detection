from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np
from sklearn.base import BaseEstimator
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from pandas import Series

# TODO X to lowercase


class NoVarianceFilter(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X if np.var(X.indices) > 0 else None


class KeySelector(BaseEstimator):
    def __init__(self, key, default_column=True):
        self.key = key
        self.default_column = default_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, Series):
            return X if self.default_column else np.zeros((len(X), 1))

        return X[self.key] if self.key in X.columns else np.zeros((len(X), 1))


class ExceptKeySelector(BaseEstimator):
    def __init__(self, key, default_column=True):
        self.key = key
        self.default_column=default_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, Series):
            return X if self.default_column else np.zeros((len(X), 1))

        return X.drop(self.key, axis=1) if len(X.columns) > 1 else np.zeros((len(X), 1))


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec=None):
        self.word2vec = word2vec
        self.dim = word2vec.vector_size

    def fit(self, X, y=None):
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


class DocEmbeddingVectorizer(object):
    def __init__(self, doc2vec):
        self.doc2vec = doc2vec

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([
            self.doc2vec.infer_vector(words) for words in X
        ])


class GensimTokenizer(BaseEstimator):
    def __init__(self, filter_stop_words):
        self.filter_stop_words = filter_stop_words
        self.stop_words = set(stopwords.words('english'))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        tokens = []
        for words in X:
            token = simple_preprocess(words.replace('\'', ''), max_len=30)
            if self.filter_stop_words:
                tokens.append(list(filter(lambda x: x not in self.stop_words, token)))
            else:
                tokens.append(token)
        return tokens


class NltkTokenizer(BaseEstimator):
    def __init__(self, filter_stop_words):
        self.filter_stop_words = filter_stop_words
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        tokens = []
        for words in X:
            token = self.tokenizer.tokenize(words.replace('\'', ''))
            if self.filter_stop_words:
                tokens.append(list(filter(lambda x: x not in self.stop_words, token)))
            else:
                tokens.append(token)
        return tokens


class EkphrasisTokenizer(BaseEstimator):
    def __init__(self, filter_stop_words):
        self.filter_stop_words = filter_stop_words
        self.stop_words = set(stopwords.words('english'))
        self.text_processor = TextPreProcessor(
            normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                'time', 'url', 'date', 'number'],
            annotate={"elongated", "repeated"},
            fix_html=True,
            segmenter="twitter",
            corrector="twitter",
            unpack_hashtags=True,
            unpack_contractions=False,
            spell_correct_elong=False,
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        tokens = []
        for words in X:
            token = self.text_processor.pre_process_doc(words.replace('\'', ''))
            if self.filter_stop_words:
                tokens.append(list(filter(lambda x: x not in self.stop_words, token)))
            else:
                tokens.append(token)
        return tokens
