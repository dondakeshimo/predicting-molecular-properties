import pandas as pd
import pickle


def load_data_from_pickle(file_folder):
    print(f"load data from {file_folder}/preprocessed/train.pickle")
    with open(f"{file_folder}/preprocessed/train.pickle", "rb") as f:
        train = pickle.load(f)

    with open(f"{file_folder}/preprocessed/test.pickle", "rb") as f:
        test = pickle.load(f)

    return train, test


def load_data_from_csv(file_folder):
    print(f"load data from {file_folder}/original/train.csv")
    train = pd.read_csv(f"{file_folder}/original/train.csv")
    test = pd.read_csv(f"{file_folder}/original/test.csv")
    structures = pd.read_csv(f"{file_folder}/original/structures.csv")
    contrib = pd.read_csv(
        f"{file_folder}/original/scalar_coupling_contributions.csv")

    return train, test, structures, contrib


def dump_data_as_pickle(train, test):
    print(f"dump data to ../data/")
    with open('../data/preprocessed/train.pickle', 'wb') as f:
        pickle.dump(train, f)

    with open('../data/preprocessed/test.pickle', 'wb') as f:
        pickle.dump(test, f)


def dump_data_as_csv(train, test):
    train.to_csv("../data/preprocessed/train.csv", index=False)
    test.to_csv("../data/preprocessed/test.csv", index=False)
