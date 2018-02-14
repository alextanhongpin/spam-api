import os
import re

from concurrent.futures import ProcessPoolExecutor

from sklearn.base import BaseEstimator, TransformerMixin

from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

class NLTKPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, strip = True, stem = True, symbols = True, stemmer = None):
        self.strip = strip
        self.stem = stem
        self.symbols = symbols
        self.stemmer = stemmer or PorterStemmer()
        self.pattern = re.compile(r'\W')
        
    def fit(self, X, y = None, **fit_params):
        return self

    def transform(self, X):
        with ProcessPoolExecutor(max_workers = os.cpu_count() * 5) as executor:
            futures = [executor.submit(self.tokenize, X_i) 
                    for X_i in X]
            X_out = [future.result() 
                    for future in futures]
            return X_out

    def tokenize(self, sentence):
        words = word_tokenize(sentence)
        out = []
        for word in words:
        if self.symbols and self.pattern.search(word):
            continue
        word = word.strip() if self.strip else word
        word = self.stemmer.stem(word) if self.stem else word
        out.append(word)
        return ' '.join(out)
