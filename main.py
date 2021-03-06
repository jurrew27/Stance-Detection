import pandas as pd
from gensim.models import KeyedVectors
from prettytable import PrettyTable
from sklearn import metrics
from stance_detector import StanceDetector
from pipelines import *


def test_classifier(train_data, test_data, pipeline, n=1):
    targets = train_data['Target'].unique()

    f1_micro = np.zeros(n)
    f1_per_target = np.zeros((len(targets), n))
    for run in range(n):

        ys_predicted = []
        ys_test = []
        for index, target in enumerate(targets):
            sd = StanceDetector(
                train_data[train_data['Target'] == target]['Tweet'],
                train_data[train_data['Target'] == target]['Stance'],
                test_data[test_data['Target'] == target]['Tweet'],
                test_data[test_data['Target'] == target]['Stance'],
            )

            y_predicted = sd.predict_model(pipeline)
            ys_predicted.extend(y_predicted)
            ys_test.extend(sd.y_test)

            result = metrics.classification_report(sd.y_test, y_predicted, output_dict=True)
            f1_per_target[index, run] = (result['FAVOR']['f1-score'] + result['AGAINST']['f1-score']) / 2.0

        result = metrics.classification_report(ys_test, ys_predicted, output_dict=True)
        f1_micro[run] = (result['FAVOR']['f1-score'] + result['AGAINST']['f1-score']) / 2.0

    t = PrettyTable(['Classifier', 'F1-score'])
    t.add_row(['f1_micro', round(np.mean(f1_micro), 3)])
    t.add_row(['f1_macro', round(np.mean(f1_per_target), 3)])
    for target, f1 in zip(targets, f1_per_target):
        t.add_row([target, round(np.mean(f1), 3)])
    print(t)

if __name__ == '__main__':
    train_data = pd.read_csv('data/train_clean.csv', escapechar='\\', encoding='latin1')
    test_data = pd.read_csv('data/test_clean.csv', escapechar='\\', encoding='latin1')
    embedding = KeyedVectors.load('embeddings_preloaded/word2vec-google-news-300.wv', mmap='r')
    # embedding = KeyedVectors.load('embeddings_preloaded/word2vec-twitter-400.wv', mmap='r')

    test_classifier(train_data, test_data, get_pipeline(
        ngram_type='tfidf',
        embedding=embedding,
        embedding_tokenizer='ekphrasis',
        pca=64
    ))
