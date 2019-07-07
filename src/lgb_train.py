import time
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    maes = (y_true - y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()


def train_lgb_model(
        X, X_test, y, params, folds, model_type="lgb", eval_metric="mae",
        plot_feature_importance=False, model=None,
        verbose=10000, early_stopping_rounds=200, n_estimators=50000):
    columns = X.columns
    result_dict = {}
    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print("--------------------------------------------------------------")
        print(f"Fold {fold_n + 1} started at {time.ctime()}")
        if type(X) == np.ndarray:
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        model = lgb.LGBMRegressor(
            **params, n_estimators=n_estimators, n_jobs=-1)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_metric=eval_metric,
            verbose=verbose,
            early_stopping_rounds=early_stopping_rounds)

        y_pred_valid = model.predict(X_valid)
        y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

        oof[valid_index] = y_pred_valid.reshape(-1,)
        scores.append(
            group_mean_log_mae(y_valid, y_pred_valid, X_valid["type"]))

        prediction += y_pred

        if model_type == "lgb" and plot_feature_importance:
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat(
                [feature_importance, fold_importance], axis=0)

    prediction /= folds.n_splits

    print("CV mean score: {0:.4f}, std: {1:.4f}.".format(
        np.mean(scores), np.std(scores)))

    result_dict["oof"] = oof
    result_dict["prediction"] = prediction
    result_dict["scores"] = scores

    if plot_feature_importance:
        feature_importance["importance"] /= folds.n_splits
        cols = (
            feature_importance[["feature", "importance"]]
            .groupby("feature").mean()
            .sort_values(by="importance", ascending=False)[:50]
            .index
        )

        best_features = feature_importance.loc[
            feature_importance.feature.isin(cols)]

        plt.figure(figsize=(16, 12))
        sns.barplot(
            x="importance", y="feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
        plt.title("LGB Features (avg over folds)")

        result_dict["feature_importance"] = feature_importance

    return result_dict
