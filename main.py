import pandas as pd
import gensim.downloader as api
from gensim.models import Word2Vec
from stance_detector import StanceDetector
from pipelines import *

train_data = pd.read_csv('data/train_clean.csv', escapechar='\\', encoding ='latin1')
test_data = pd.read_csv('data/test_clean.csv', escapechar='\\', encoding ='latin1')

parameters = {
    'features__d2v__vectorizer__iter': [2, 3, 4, 5, 6, 7, 8, 9, 10],
}

embedding = api.load('word2vec-google-news-300')
# embedding = Word2Vec.load('embeddings/word2vec-sg-300.model')

sd = StanceDetector(train_data['Tweet'], train_data['Stance'], test_data['Tweet'], test_data['Stance'])

# sd.grid_search_model(get_tfidf_ngram_dm_embedding_pipeline(), parameters)

y_predicted = sd.predict_model(get_mean_embedding_pipeline(embedding))
sd.analyze_classifier_results(y_predicted)

y_predicted = sd.predict_model(get_tfidf_embedding_pipeline(embedding))
sd.analyze_classifier_results(y_predicted)