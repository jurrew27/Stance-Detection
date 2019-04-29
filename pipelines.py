from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from feature_extractors import *

# TODO add support for features that should not be tokenized

def get_ngram_pipeline(ngram_type):
    if ngram_type == 'binary':
        return FeatureUnion([
            ('chars', CountVectorizer(ngram_range=(2, 5), analyzer='char',
                strip_accents='unicode', binary=True)),
            ('words', CountVectorizer(ngram_range=(1, 3), analyzer='word',
                stop_words='english', strip_accents='unicode', binary=True))
        ])
    elif ngram_type == 'count':
        return FeatureUnion([
            ('chars', CountVectorizer(ngram_range=(2, 5), analyzer='char',
                strip_accents='unicode')),
            ('words', CountVectorizer(ngram_range=(1, 3), analyzer='word',
                stop_words='english', strip_accents='unicode'))
        ])
    elif ngram_type == 'tfidf':
        return FeatureUnion([
            ('chars', TfidfVectorizer(ngram_range=(2, 5), analyzer='char',
                sublinear_tf=True, strip_accents='unicode')),
            ('words', TfidfVectorizer(ngram_range=(1, 3), analyzer='word',
                sublinear_tf=True, stop_words='english', strip_accents='unicode'))
        ])
    else:
        return Drop()


def get_embedding_pipeline(tweet_tokenizer, vectorizer, embedding, pca):
    if vectorizer == 'mean':
        vectorizer = MeanEmbeddingVectorizer
    elif vectorizer == 'tfidf':
        vectorizer = TfidfEmbeddingVectorizer
    elif vectorizer == 'doc':
        vectorizer = DocEmbeddingVectorizer

    return Pipeline([
        ('tokenizer', TweetSplitter() if tweet_tokenizer else SentenceSplitter()),
        ('vectorizer', vectorizer(embedding)),
        ('pca', PCA(n_components=25) if pca else PassThrough())
    ])


def get_pipeline(
        ngram='count',
        tweet_tokenizer=False,
        embedding_vectorizer='mean',
        embedding=None,
        pca=False
):

    if embedding:
        embedding_pl = get_embedding_pipeline(tweet_tokenizer, embedding_vectorizer, embedding, pca)
    else:
        embedding_pl = Drop()

    return [
        ('features', FeatureUnion([
            ('ngrams', get_ngram_pipeline(ngram)),
            ('embedding', embedding_pl)
        ])),
        ('clf', LinearSVC(max_iter=10000))
    ]
