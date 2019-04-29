import pandas as pd
import gensim.downloader as api
from gensim.models import Word2Vec, Doc2Vec
from sklearn import metrics
from stance_detector import StanceDetector
from pipelines import *


def test_classifier(train_data, test_data, pipeline):
    targets = train_data['Target'].unique()

    f1_averages = []
    ys_predicted = []
    ys_test = []
    for target in targets:
        sd = StanceDetector(
            train_data[train_data['Target'] == target][['Tweet', 'Opinion Towards']],
            train_data[train_data['Target'] == target]['Stance'],
            test_data[test_data['Target'] == target][['Tweet', 'Opinion Towards']],
            test_data[test_data['Target'] == target]['Stance'],
        )

        y_predicted = sd.predict_model(pipeline)
        ys_predicted.extend(y_predicted)
        ys_test.extend(sd.y_test)

        result = metrics.classification_report(sd.y_test, y_predicted, output_dict=True)
        f1_averages.append((result['FAVOR']['f1-score'] + result['AGAINST']['f1-score']) / 2.0)

    result = metrics.classification_report(ys_test, ys_predicted, output_dict=True)
    f1_micro = (result['FAVOR']['f1-score'] + result['AGAINST']['f1-score']) / 2.0
    f1_macro = np.mean(f1_averages)

    # TODO nicely format this
    print('-----------------------------------')
    for target, f1 in zip(targets, f1_averages):
        print('{}: {}'.format(target, f1))
    print('f1_micro: {}'.format(f1_micro))
    print('f1_macro: {}'.format(f1_macro))
    print('-----------------------------------')

if __name__ == '__main__':
    train_data = pd.read_csv('data/train_clean.csv', escapechar='\\', encoding='latin1')
    test_data = pd.read_csv('data/test_clean.csv', escapechar='\\', encoding='latin1')
    # embedding = api.load('word2vec-google-news-300')
    # embedding = Word2Vec.load('embeddings/word2vec-sg-300-all.model')
    test_classifier(train_data, test_data, get_pipeline(
        ngram='binary'
    ))
    test_classifier(train_data, test_data, get_pipeline(
        ngram='count'
    ))
    test_classifier(train_data, test_data, get_pipeline(
        ngram='tfidf'
    ))
