def loader(file_input):
    '''loads data from the enron dataset

    e.g. 
    ham = loader('../data/enron1/ham')
    spam = loader('../data/enron1/spam')
    '''
    data = []
    for (dirpath, dirnames, filenames) in os.walk(file_input):
        for file in filenames:
            path = os.path.join(dirpath, file)
            with open(path, encoding='latin-1') as f:
                data.append(f.read())
    return data

