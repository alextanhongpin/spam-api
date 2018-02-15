import sys
from flask import Flask
from sklearn.externals import joblib
from pipeline.array_transformer import ArrayTransformer
from pipeline.nltk_preprocessor import NLTKPreprocessor

# sys.modules['NLTKPreprocessor'] = NLTKPreprocessor

app = Flask(__name__)
clf = joblib.load('models/gaussian_nb.pkl')
print(clf)

@app.route('/')
def hello():
  # NLTKPreprocessor.__module__ = 'NLTKPreprocessor'

  text = ['hello world']
  print('got prediction for', clf.predict(text))
  return 'hello world'

# NOTE: This part is only needed if you run the application with python -m main
# If you run it with FLASK_APP=main.py flask run, this part will not be executed
if __name__ == '__main__':
  app.run(debug = True, host = '0.0.0.0')