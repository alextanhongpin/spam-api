def linear_svc_clf():
    from sklearn.svm import LinearSVC
    
    estimators = [('linear_svc', LinearSVC())]
    param_grid = dict(linear_svc__C = [1, 10, 100],
                      linear_svc__random_state = [42],
                      linear_svc__max_iter = [1000, 10000])

    return estimators, param_grid

def multinomial_nb_clf():
    from sklearn.naive_bayes import MultinomialNB
    
    estimators = [('multinomial_nb', MultinomialNB())]
    param_grid = dict(multinomial_nb__alpha = [1, 10, 100],
                      multinomial_nb__fit_prior = [True],
                      multinomial_nb__class_prior = [None])

    return estimators, param_grid

def random_forest_clf():
    from sklearn.ensemble import RandomForestClassifier

    estimators = [('random_forest', RandomForestClassifier())]
    param_grid = dict(random_forest__n_estimators = [10, 20, 30],
                      random_forest__criterion = ['gini', 'entropy'],
                      random_forest__max_features = ['auto', 'sqrt', 'log2'],
#                       random_forest__max_depth = [None],
                      random_forest__min_samples_split = [2],
                      random_forest__min_samples_leaf = [1],
                      random_forest__min_weight_fraction_leaf = [0],
                      random_forest__max_leaf_nodes = [None],
                      random_forest__min_impurity_decrease = [0],
                      random_forest__bootstrap = [True],
                      random_forest__oob_score = [False],
                      random_forest__n_jobs = [-1],
                      random_forest__random_state = [42],
                      random_forest__warm_start = [False],
                      random_forest__class_weight = ['balanced'])

    return estimators, param_grid

def gaussian_nb_clf():
    from sklearn.naive_bayes import GaussianNB
    
    estimators = [('gaussian_nb', GaussianNB())]
    param_grid = dict()
    return estimators, param_grid