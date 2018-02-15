import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

from utils.csv import to_csv, read_csv
from pipeline.array_transformer import ArrayTransformer
from pipeline.nltk_preprocessor import NLTKPreprocessor
from classifiers import linear_svc_clf, multinomial_nb_clf, random_forest_clf, gaussian_nb_clf

if __name__ == '__main__':
    # Load data
    X, y = read_csv('data.csv')
    print(X.shape, y.shape)

    # Split training/testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
    clfs = [
        linear_svc_clf(),
        multinomial_nb_clf(),
        random_forest_clf(),
        gaussian_nb_clf()
    ]

    labels = ['0', '1']

    def build_model(estimators, param_grid):
        clf_name = estimators[0][0]
        print('Start')
        print('Classifier name: {}'.format(clf_name))
        
        # Avoid error when pickling
        #   NLTKPreprocessor.__module__ = 'pipeline_nltk_preprocessor'
        #   ArrayTransformer.__module__ = 'pipeline_array_transformer'

        pipeline_estimators = [('nltk_preprocessor', NLTKPreprocessor()),
                                ('vectorizer', CountVectorizer(stop_words = 'english')),
                                ('tfidf', TfidfTransformer()),
                                ('transformer', ArrayTransformer()),
                                estimators[0]]
        
        pipeline = Pipeline(pipeline_estimators)
        clf = GridSearchCV(pipeline, param_grid = param_grid)
        clf.fit(X_train, y_train)
        
        print('Best params:\n', clf.best_params_, '\n')
        print('Classifier results:\n', clf.cv_results_, '\n')
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_true = y_test, 
                              y_pred = y_pred, 
                              labels = labels)
        print('Confusion matrix:\n', cm)

        report = classification_report(y_true = y_test, 
                                       y_pred = y_pred, 
                                       target_names = labels)
        print('Classification report:\n', report)
        print('End')
        return clf_name, clf

    results = [build_model(estimators, param_grid) 
               for (estimators, param_grid) in clfs]

    for (clf_name, clf) in results:
        joblib.dump(clf, '../models/{}.pkl'.format(clf_name)) 
        print('model saved as ../models/{}.pkl'.format(clf_name))
