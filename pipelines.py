from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from feature_extractors import *


def get_ngram_pipeline(ngram_type, char_ngrams, word_ngrams):
    if ngram_type == 'binary':
        return FeatureUnion([
            ('chars', CountVectorizer(ngram_range=char_ngrams, analyzer='char',
                strip_accents='unicode', binary=True)),
            ('words', CountVectorizer(ngram_range=word_ngrams, analyzer='word',
                stop_words='english', strip_accents='unicode', binary=True))
        ])
    elif ngram_type == 'count':
        return FeatureUnion([
            ('chars', CountVectorizer(ngram_range=char_ngrams, analyzer='char',
                strip_accents='unicode')),
            ('words', CountVectorizer(ngram_range=word_ngrams, analyzer='word',
                stop_words='english', strip_accents='unicode'))
        ])
    elif ngram_type == 'tfidf':
        return FeatureUnion([
            ('chars', TfidfVectorizer(ngram_range=char_ngrams, analyzer='char',
                sublinear_tf=True, strip_accents='unicode')),
            ('words', TfidfVectorizer(ngram_range=word_ngrams, analyzer='word',
                sublinear_tf=True, stop_words='english', strip_accents='unicode'))
        ])
    else:
        return None


def get_embedding_pipeline(tokenizer, vectorizer, embedding, pca, filter_stop_words):
    if vectorizer == 'mean':
        vectorizer = MeanEmbeddingVectorizer
    elif vectorizer == 'tfidf':
        vectorizer = TfidfEmbeddingVectorizer
    elif vectorizer == 'doc':
        vectorizer = DocEmbeddingVectorizer

    if tokenizer == 'nltk':
        tokenizer = NltkTokenizer
    elif tokenizer == 'ekphrasis':
        tokenizer = EkphrasisTokenizer
    elif tokenizer == 'gensim':
        tokenizer = GensimTokenizer

    if pca:
        return Pipeline([
            ('tokenizer', tokenizer(filter_stop_words=filter_stop_words)),
            ('vectorizer', vectorizer(embedding)),
            ('pca', PCA(n_components=pca))
        ])
    else:
        return Pipeline([
            ('tokenizer', tokenizer(filter_stop_words=filter_stop_words)),
            ('vectorizer', vectorizer(embedding))
        ])


def get_pipeline(
        ngram_type='tfidf',
        char_ngrams=(2, 5),
        word_ngrams=(1, 3),
        embedding_tokenizer='ekphrasis',
        embedding_vectorizer='mean',
        embedding=None,
        pca=False,
        filter_stop_words=False
):

    if embedding:
        embedding_pl = get_embedding_pipeline(embedding_tokenizer, embedding_vectorizer, embedding, pca, filter_stop_words)
    else:
        embedding_pl = None

    if ngram_type:
        ngram_pl = get_ngram_pipeline(ngram_type, char_ngrams, word_ngrams)
    else:
        ngram_pl = None

    return [
        ('features', ColumnTransformer([
            ('tweet', FeatureUnion([
                ('ngrams', ngram_pl),
                ('embedding', embedding_pl)
             ]), 'Tweet')
        ], remainder=OneHotEncoder())),
        ('clf', LinearSVC(max_iter=10000))
    ]

def get_simple_pipeline(
        ngram_type='tfidf',
        char_ngrams=(2, 5),
        word_ngrams=(1, 3),
        embedding_tokenizer='ekphrasis',
        embedding_vectorizer='mean',
        embedding=None,
        pca=False,
        filter_stop_words=False
):

    if embedding:
        embedding_pl = get_embedding_pipeline(embedding_tokenizer, embedding_vectorizer, embedding, pca, filter_stop_words)
    else:
        embedding_pl = None

    if ngram_type:
        ngram_pl = get_ngram_pipeline(ngram_type, char_ngrams, word_ngrams)
    else:
        ngram_pl = None

    return [
        ('tweet', FeatureUnion([
            ('ngrams', ngram_pl),
            ('embedding', embedding_pl)
        ])),
        ('clf', LinearSVC(max_iter=10000))
    ]
