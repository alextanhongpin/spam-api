# Spam API

Microservices API for spam filtering system.

## Abstract

One of the goals of this repository is design an approach to design machine learning systems.

## To run 

```bash
$ python -m main

# Note that this will not work since the import will be messed up
$ python main.py
```

## Flows

- Prepare text data
  - removal of stop words
  - lemmatization
- Feature extraction process
- Training the classifiers
- Checking performance


## Pickled

To view the size of the pickled file:
```bash
$ du -h *.pkl
```

## Tips

At first it may be tempting to construct your pipeline to include the feature extractor:

```python
pipeline = Pipeline([('vect', CountVectorizer(stop_words = 'english')),
                      ('tfidf', TfidfTransformer()),
                      ('gaussian_nb', GaussianNB())])
```

But note that this will only be useful when training your model. For prediction, you need to reuse the feature extractor function. Also,
when training multiple classifiers, you will end up running the feature extraction process which is not optimal.
