import pandas as pd

def to_csv(file_name, features, labels):
    df = pd.DataFrame({'features': features,
                       'labels': labels})
    df.to_csv('{}'.format(file_name))
    print('Wrote to {}'.format(file_name))

def read_csv(file_name):
    df = pd.read_csv(file_name)
    X = df.as_matrix(columns = ['features']).flatten()
    y = df.as_matrix(columns = ['labels']).astype(str).flatten()
    return X, y