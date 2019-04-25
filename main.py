import pandas as pd
import gensim.downloader as api
from gensim.models import Word2Vec
from stance_detector import StanceDetector
from pipelines import *

train_data = pd.read_csv('data/train_clean.csv', escapechar='\\', encoding ='latin1')
test_data = pd.read_csv('data/test_clean.csv', escapechar='\\', encoding ='latin1')

parameters = {
    'clf__loss': ['hinge', 'squared_hinge'],
    # 'clf__tol': [1e-4, 1e-3, 1e-2],
    'clf__C': [1, 10, 100],
    'clf__fit_intercept': [True, False]
}

embedding = api.load('word2vec-google-news-300')
# embedding = Word2Vec.load('embeddings/word2vec_cbow-200.model')

sd = StanceDetector(train_data['Tweet'], train_data['Stance'], test_data['Tweet'], test_data['Stance'])

# sd.grid_search_model(get_mean_embedding_pipeline(glove_twitter), parameters)

y_predicted = sd.predict_model(get_tfidf_ngram_tfidf_embedding_pipeline(embedding))
sd.analyze_classifier_results(y_predicted)

