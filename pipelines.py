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
        ngram_type='count',
        char_ngrams=(2, 5),
        word_ngrams=(1, 3),
        tweet_tokenizer=False,
        embedding_vectorizer='mean',
        embedding=None,
        pca=False
):

    if embedding:
        embedding_pl = get_embedding_pipeline(tweet_tokenizer, embedding_vectorizer, embedding, pca)
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