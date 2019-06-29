import pandas as pd
import pickle


def load_data_from_pickle(file_folder):
    print(f"load data from {file_folder}/train.pickle")
    with open(f"{file_folder}/train.pickle", "rb") as f:
        train = pickle.load(f)

    with open(f"{file_folder}/test.pickle", "rb") as f:
        test = pickle.load(f)

    return train, test


def load_data_from_csv(file_folder):
    print(f"load data from {file_folder}/train.csv")
    train = pd.read_csv(f"{file_folder}/train.csv")
    test = pd.read_csv(f"{file_folder}/test.csv")
    structures = pd.read_csv(f"{file_folder}/structures.csv")
    contrib = pd.read_csv(f"{file_folder}/scalar_coupling_contributions.csv")

    return train, test, structures, contrib


def dump_data_as_pickle(train, test):
    with open('../data/train.pickle', 'wb') as f:
        pickle.dump(train, f)

    with open('../data/test.pickle', 'wb') as f:
        pickle.dump(test, f)
