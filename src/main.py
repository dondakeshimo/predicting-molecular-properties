import artgor_utils
import nn_train
import os
import pandas as pd
import pickle
import preprocess
from sklearn.model_selection import KFold


def load_n_preprocess_data(file_folder, init_flag=False):
    if not init_flag and \
       os.path.exists(f"{file_folder}/train.pickle") and \
       os.path.exists(f"{file_folder}/train.pickle"):
        return load_data_from_pickle(file_folder)
    else:
        train, test, structures, contrib = load_data_from_csv(file_folder)
        train, test = preprocess_data(train, test, structures, contrib)
        dump_data_as_pickle(train, test)
        return train, test


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


def preprocess_data(train, test, structures, contrib):
    train = pd.merge(train, contrib, how="left",
                     left_on=["molecule_name", "atom_index_0",
                              "atom_index_1", "type"],
                     right_on=["molecule_name", "atom_index_0",
                               "atom_index_1", "type"])

    structures = preprocess.get_atom_rad_en(structures)
    structures = preprocess.calc_bonds(structures)

    train = preprocess.map_atom_info(train, structures, 0)
    train = preprocess.map_atom_info(train, structures, 1)
    test = preprocess.map_atom_info(test, structures, 0)
    test = preprocess.map_atom_info(test, structures, 1)

    train = preprocess.calc_dist(train)
    test = preprocess.calc_dist(test)

    train["type_0"] = train["type"].apply(lambda x: x[0])
    test["type_0"] = test["type"].apply(lambda x: x[0])

    train = preprocess.create_features(train)
    test = preprocess.create_features(test)

    train, test = preprocess.encode_str(train, test)

    return train, test


def dump_data_as_pickle(train, test):
    with open('../data/train.pickle', 'wb') as f:
        pickle.dump(train, f)

    with open('../data/test.pickle', 'wb') as f:
        pickle.dump(test, f)


def train_each_type_with_lgb(X, X_test, y, folds):
    params = {"num_leaves": 128,
              "min_child_samples": 79,
              "objective": "regression",
              "max_depth": 9,
              "learning_rate": 0.2,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 0.9,
              "bagging_seed": 11,
              "metric": "mae",
              "verbosity": -1,
              "reg_alpha": 0.1,
              "reg_lambda": 0.3,
              "colsample_bytree": 1.0
              }

    X_short = pd.DataFrame(
        {"ind": list(X.index),
         "type": X["type"].values,
         "oof": [0] * len(X),
         "target": y.values})
    X_short_test = pd.DataFrame(
        {"ind": list(X_test.index),
         "type": X_test["type"].values,
         "prediction": [0] * len(X_test)})

    for t in X["type"].unique():
        print(f"Training of type {t}")
        X_t = X.loc[X["type"] == t]
        X_test_t = X_test.loc[X_test["type"] == t]
        y_t = X_short.loc[X_short["type"] == t, "target"]
        result_dict_lgb = artgor_utils.train_model_regression(
            X=X_t, X_test=X_test_t, y=y_t, params=params, folds=folds,
            model_type="lgb", eval_metric="group_mae",
            plot_feature_importance=False,
            verbose=5, early_stopping_rounds=5, n_estimators=10)
        X_short.loc[X_short["type"] == t, "oof"] = result_dict_lgb["oof"]
        X_short_test.loc[X_short_test["type"] == t, "prediction"] = \
            result_dict_lgb["prediction"]

    return X_short, X_short_test


def train_each_type_with_nn(X, X_test, y, folds):
    X_short = pd.DataFrame(
        {"ind": list(X.index),
         "type": X["type"].values,
         "oof": [0] * len(X),
         "target": y.values})
    X_short_test = pd.DataFrame(
        {"ind": list(X_test.index),
         "type": X_test["type"].values,
         "prediction": [0] * len(X_test)})

    for t in X["type"].unique():
        print(f"Training of type {t}")
        X_t = X.loc[X["type"] == t]
        X_test_t = X_test.loc[X_test["type"] == t]
        y_t = X_short.loc[X_short["type"] == t, "target"]
        result_dict_lgb = nn_train.train_nn_model(
            X=X_t, X_test=X_test_t, y=y_t, folds=folds,
            verbose=2, epochs=10, batch_size=32)
        X_short.loc[X_short["type"] == t, "oof"] = result_dict_lgb["oof"]
        X_short_test.loc[X_short_test["type"] == t, "prediction"] = \
            result_dict_lgb["prediction"]

    return X_short, X_short_test


def main():
    file_folder = "../data"
    train, test = load_n_preprocess_data(file_folder, init_flag=False)

    good_columns = preprocess.get_good_columns()

    X = train[good_columns].copy()
    y = train["scalar_coupling_constant"]
    y_fc = train["fc"]
    X_test = test[good_columns].copy()

    n_fold = 3
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)

    X_short, X_short_test = train_each_type_with_lgb(X, X_test, y_fc, folds)

    X["oof_fc"] = X_short["oof"]
    X_test["oof_fc"] = X_short_test["prediction"]

    X_lgb, X_lgb_test = \
        train_each_type_with_lgb(X, X_test, y, folds)

    X_nn, X_nn_test = \
        train_each_type_with_nn(X, X_test, y, folds)

    submit = (X_lgb_test["prediction"] + X_nn_test["prediction"]) / 2

    sub = pd.read_csv(f"{file_folder}/sample_submission.csv")
    sub["scalar_coupling_constant"] = submit
    sub.to_csv(f"{file_folder}/submission.csv", index=False)
    sub.head()


if __name__ == '__main__':
    main()
