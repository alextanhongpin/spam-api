from sklearn.base import BaseEstimator, TransformerMixin

class ArrayTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None, **fit_params):
        return self

    def transform(self, X):
        return X.toarray()