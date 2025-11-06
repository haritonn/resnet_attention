import pickle

def load_cifar100_file(filename):
    with open(filename, 'rb') as f:
        data_dict = pickle.load(filename)

    return data_dict

