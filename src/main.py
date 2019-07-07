import argparse
import os
import pandas as pd
import preprocess
from sklearn.model_selection import KFold

import handle_files
import lgb_train
import nn_train


def load_n_preprocess_data(file_folder, init_flag=False):
    if not init_flag and \
       os.path.exists(f"{file_folder}/preprocessed/train.pickle") and \
       os.path.exists(f"{file_folder}/preprocessed/train.pickle"):
        return handle_files.load_data_from_pickle(file_folder)
    else:
        train, test, structures, contrib = \
            handle_files.load_data_from_csv(file_folder)
        train, test = preprocess.preprocess(train, test, structures, contrib)
        handle_files.dump_data_as_pickle(train, test)
        return train, test


def train_full_with_lgb(X, X_test, y, folds):
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

    result_dict_lgb = lgb_train.train_lgb_model(
        X=X, X_test=X_test, y=y, params=params, folds=folds, model_type="lgb",
        eval_metric="group_mae", plot_feature_importance=True,
        verbose=300, early_stopping_rounds=1000, n_estimators=3000)

    return result_dict_lgb


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
        print("==============================================================")
        print(f"Training of type {t}")
        X_t = X.loc[X["type"] == t]
        X_test_t = X_test.loc[X_test["type"] == t]
        y_t = X_short.loc[X_short["type"] == t, "target"]
        result_dict_lgb = lgb_train.train_lgb_model(
            X=X_t, X_test=X_test_t, y=y_t, params=params, folds=folds,
            model_type="lgb", eval_metric="group_mae",
            plot_feature_importance=False,
            verbose=500, early_stopping_rounds=200, n_estimators=4000)
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
        X_t, X_test_t = nn_train.fit_scale_data(X_t, X_test_t)
        y_t = X_short.loc[X_short["type"] == t, "target"].values
        result_dict_lgb = nn_train.train_nn_model(
            X=X_t, X_test=X_test_t, y=y_t, folds=folds,
            verbose=1, epochs=10, batch_size=32)
        X_short.loc[X_short["type"] == t, "oof"] = result_dict_lgb["oof"]
        X_short_test.loc[X_short_test["type"] == t, "prediction"] = \
            result_dict_lgb["prediction"]

    return X_short, X_short_test


def main_importance(args):
    file_folder = args.input_dir
    train, test, structures, contrib = \
        handle_files.load_data_from_csv(file_folder)
    train, test = preprocess.create_feature_importance(
        train, test, structures, contrib)

    full_columns = train.drop(
        ["id", "scalar_coupling_constant", "molecule_name",
         "fc", "dso", "sd", "pso"], axis=1).columns

    X = train[full_columns].copy()
    y = train["scalar_coupling_constant"]
    X_test = test[full_columns].copy()

    del train, test

    n_fold = 3
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)

    result_dict_lgb = train_full_with_lgb(X, X_test, y, folds)

    result_dict_lgb["feature_importance"].to_csv(
        f"{file_folder}/preprocessed/feature_importance.csv", index=False)


def main_fc(args):
    file_folder = args.input_dir
    init_flag = args.init_pickle_flag
    train, test = load_n_preprocess_data(file_folder, init_flag)

    good_columns = preprocess.get_good_columns()

    X = train[good_columns].copy()
    y_fc = train["fc"]
    X_test = test[good_columns].copy()

    del train, test

    n_fold = 3
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)

    X_fc, X_fc_test = train_each_type_with_lgb(X, X_test, y_fc, folds)

    df_train_oof_fc = pd.DataFrame({"train_oof_fc": X_fc["oof"]})
    df_test_oof_fc = pd.DataFrame({"test_oof_fc": X_fc_test["prediction"]})

    df_train_oof_fc.to_csv(
        f"{file_folder}/preprocessed/train_oof_fc.csv", index=False)
    df_test_oof_fc.to_csv(
        f"{file_folder}/preprocessed/test_oof_fc.csv", index=False)


def main_lgb(args):
    file_folder = args.input_dir
    init_flag = args.init_pickle_flag
    train, test = load_n_preprocess_data(file_folder, init_flag)

    good_columns = preprocess.get_good_columns()

    X = train[good_columns].copy()
    y = train["scalar_coupling_constant"]
    X_test = test[good_columns].copy()

    del train, test

    if args.oof_fc_flag:
        df_train_oof_fc = pd.read_csv(
            f"{file_folder}/preprocessed/train_oof_fc.csv")
        df_test_oof_fc = pd.read_csv(
            f"{file_folder}/preprocessed/test_oof_fc.csv")
        X["oof_fc"] = df_train_oof_fc["train_oof_fc"].values
        X_test["oof_fc"] = df_test_oof_fc["test_oof_fc"].values

    n_fold = 3
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)

    X_lgb, X_lgb_test = \
        train_each_type_with_lgb(X, X_test, y, folds)

    df_train_oof_lgb = pd.DataFrame({"train_oof_lgb": X_lgb["oof"]})
    df_lgb_prediction = pd.DataFrame(
        {"lgb_prediction": X_lgb_test["prediction"]})

    df_train_oof_lgb.to_csv(
        f"{file_folder}/predicted/train_oof_lgb.csv", index=False)
    df_lgb_prediction.to_csv(
        f"{file_folder}/predicted/lgb_prediction.csv", index=False)


def main_nn(args):
    file_folder = args.input_dir
    init_flag = args.init_pickle_flag
    train, test = load_n_preprocess_data(file_folder, init_flag)

    good_columns = preprocess.get_good_columns()

    X = train[good_columns].copy()
    y = train["scalar_coupling_constant"]
    X_test = test[good_columns].copy()

    del train, test

    if args.oof_fc_flag:
        df_train_oof_fc = pd.read_csv(
            f"{file_folder}/preprocessed/train_oof_fc.csv")
        df_test_oof_fc = pd.read_csv(
            f"{file_folder}/preprocessed/test_oof_fc.csv")
        X["oof_fc"] = df_train_oof_fc["train_oof_fc"].values
        X_test["oof_fc"] = df_test_oof_fc["test_oof_fc"].values

    n_fold = 3
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)

    X_nn, X_nn_test = \
        train_each_type_with_nn(X, X_test, y, folds)

    df_train_oof_nn = pd.DataFrame({"train_oof_nn": X_nn["oof"]})
    df_nn_prediction = pd.DataFrame(
        {"nn_prediction": X_nn_test["prediction"]})

    df_train_oof_nn.to_csv(
        f"{file_folder}/predicted/train_oof_nn.csv", index=False)
    df_nn_prediction.to_csv(
        f"{file_folder}/predicted/nn_prediction.csv", index=False)


def main(args):
    if args.mode.upper() == "IMPORTANCE":
        print("RUN: create feature importance mode")
        main_importance(args)

    elif args.mode.upper() == "FC":
        print("RUN: train target fc mode")
        main_fc(args)

    elif args.mode.upper() == "LGB":
        print("RUN: train with LGB mode")
        main_lgb(args)

    elif args.mode.upper() == "NN":
        print("RUN: train with NN mode")
        main_nn(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="predict molecular properties")
    parser.add_argument("-m", "--mode", help="nn", default="nn")
    parser.add_argument("-i", "--input_dir", help="../data", default="../data")
    parser.add_argument("--init_pickle_flag", action="store_true")
    parser.add_argument("--nn_epochs", default=20)
    parser.add_argument("--lgb_estimators", default=20)
    parser.add_argument("--oof_fc_flag", action="store_true")

    main(parser.parse_args())
