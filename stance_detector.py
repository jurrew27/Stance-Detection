from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


class StanceDetector:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def predict_model(self, pipeline_config):
        pipeline = Pipeline(pipeline_config)
        pipeline.fit(self.x_train, self.y_train)
        return pipeline.predict(self.x_test)

    def grid_search_model(self, pipeline_config, parameters):
        pipeline = Pipeline(pipeline_config)
        grid_search = GridSearchCV(pipeline, parameters, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(self.x_train, self.y_train)

        for i in range(len(grid_search.cv_results_['params'])):
            print(i, 'params - %s; mean - %0.3f; std - %0.3f #%d'
                  % (grid_search.cv_results_['params'][i],
                     grid_search.cv_results_['mean_test_score'][i],
                     grid_search.cv_results_['std_test_score'][i],
                     grid_search.cv_results_['rank_test_score'][i]))

    def analyze_classifier_results(self, y_predicted):
        print(metrics.classification_report(self.y_test, y_predicted, digits=3))
        print(metrics.confusion_matrix(self.y_test, y_predicted))
