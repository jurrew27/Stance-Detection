import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

train_data = pd.read_csv('data/train_clean.csv', escapechar='\\', encoding ='latin1')
test_data = pd.read_csv('data/test_clean.csv', escapechar='\\', encoding ='latin1')

features = FeatureUnion([
    ('chars', TfidfVectorizer(ngram_range=(1, 1), analyzer='char',
        use_idf=False, stop_words='english', strip_accents='unicode')),
    ('words', TfidfVectorizer(ngram_range=(1, 1), analyzer='word',
        use_idf=False, stop_words='english', strip_accents='unicode'))
])

pipeline = Pipeline([
    ('features', features),
    ('clf', LinearSVC(C=1)),
])

parameters = {
    'features__chars__ngram_range': [(2, 2), (2, 3), (2, 4), (2, 5)],
    'features__words__ngram_range': [(1, 1), (1, 2), (1, 3)]
}
grid_search = GridSearchCV(pipeline, parameters, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(train_data['Tweet'], train_data['Stance'])

n_candidates = len(grid_search.cv_results_['params'])
for i in range(n_candidates):
    print(i, 'params - %s; mean - %0.2f; std - %0.2f'
          % (grid_search.cv_results_['params'][i],
             grid_search.cv_results_['mean_test_score'][i],
             grid_search.cv_results_['std_test_score'][i]))

print(grid_search.cv_results_['rank_test_score'])

stance_predicted = grid_search.predict(test_data['Tweet'])

print(metrics.classification_report(test_data['Stance'], stance_predicted))

cm = metrics.confusion_matrix(test_data['Stance'], stance_predicted)
print(cm)