import pickle

def load_pickle(filename):
    model = pickle.load(open(filename, 'rb'))
    return model

def save_pickle(model, filename):
    # save the model to disk
    pickle.dump(model, open(filename, 'wb'))