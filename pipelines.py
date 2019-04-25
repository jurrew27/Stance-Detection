from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from gensim.sklearn_api import W2VTransformer
from feature_extractors import *


ngrams = FeatureUnion([
    ('chars', CountVectorizer(ngram_range=(2, 5), analyzer='char',
        stop_words='english', strip_accents='unicode')),
    ('words', CountVectorizer(ngram_range=(1, 3), analyzer='word',
        stop_words='english', strip_accents='unicode'))
])
tfidf_ngrams = FeatureUnion([
    ('chars', TfidfVectorizer(ngram_range=(2, 5), analyzer='char',
        sublinear_tf=True, stop_words='english', strip_accents='unicode')),
    ('words', TfidfVectorizer(ngram_range=(1, 3), analyzer='word',
        sublinear_tf=True, stop_words='english', strip_accents='unicode'))
])


def get_count_pipeline():
    return [
        ('vect', CountVectorizer()),
        ('clf', LinearSVC(max_iter=10000))
    ]


def get_ngram_pipeline() :
    return [
        ('features', ngrams),
        ('clf', LinearSVC(max_iter=10000))
    ]


def get_tfidf_ngram_pipeline():
    return [
        ('features', tfidf_ngrams),
        ('clf', LinearSVC(max_iter=10000))
    ]


def get_mean_embedding_pipeline(embedding):
    return [
        ('vect', MeanEmbeddingVectorizer(embedding)),
        ('scal', StandardScaler()),
        ('clf', LinearSVC(max_iter=10000))
    ]

def get_tfidf_embedding_pipeline(embedding):
    return [
        ('vect', TfidfEmbeddingVectorizer(embedding)),
        ('scal', StandardScaler()),
        ('clf', LinearSVC(max_iter=10000))
    ]


def get_tfidf_ngram_mean_embedding_pipeline(embedding):
    return [
        ('features', FeatureUnion([
            ('ngrams', tfidf_ngrams),
            ('vect', MeanEmbeddingVectorizer(embedding)),
        ])),
        ('clf', LinearSVC(max_iter=10000))
    ]

def get_tfidf_ngram_tfidf_embedding_pipeline(embedding):
    return [
        ('features', FeatureUnion([
            ('ngrams', tfidf_ngrams),
            ('vect', TfidfEmbeddingVectorizer(embedding)),
        ])),
        ('clf', LinearSVC(max_iter=10000))
    ]

def get_scaled_tfidf_ngram_mean_embedding_pipeline(embedding):
    return [
        ('features', FeatureUnion([
            ('ngrams', tfidf_ngrams),
            ('vect_scal', Pipeline([
                ('vect', MeanEmbeddingVectorizer(embedding)),
                ('scal', StandardScaler())
            ]))
        ])),
        ('clf', LinearSVC(max_iter=10000))
    ]

def get_scaled_tfidf_ngram_tfidf_embedding_pipeline(embedding):
    return [
        ('features', FeatureUnion([
            ('ngrams', tfidf_ngrams),
            ('vect_scal', Pipeline([
                ('vect', TfidfEmbeddingVectorizer(embedding)),
                ('scal', StandardScaler())
            ]))
        ])),
        ('clf', LinearSVC(max_iter=10000))
    ]