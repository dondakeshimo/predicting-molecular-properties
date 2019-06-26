import artgor_utils
import pandas as pd
import utils
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder


file_folder = "../data"

train = pd.read_csv(f"{file_folder}/train.csv")
test = pd.read_csv(f"{file_folder}/test.csv")
sub = pd.read_csv(f"{file_folder}/sample_submission.csv")
structures = pd.read_csv(f"{file_folder}/structures.csv")

structures = utils.get_atom_rad_en(structures)
structures = utils.calc_bonds(structures)

train = utils.map_atom_info(train, structures, 0)
train = utils.map_atom_info(train, structures, 1)
test = utils.map_atom_info(test, structures, 0)
test = utils.map_atom_info(test, structures, 1)

train = utils.calc_dist(train)
test = utils.calc_dist(test)

train["type_0"] = train["type"].apply(lambda x: x[0])
test["type_0"] = test["type"].apply(lambda x: x[0])

train = utils.create_features(train)
test = utils.create_features(test)

good_columns = utils.get_good_columns()

for f in ["atom_0", "atom_1", "type_0", "type"]:
    if f in good_columns:
        lbl = LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

X = train[good_columns].copy()
y = train["scalar_coupling_constant"]
X_test = test[good_columns].copy()

n_fold = 3
folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)

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

# result_dict_lgb = artgor_utils.train_model_regression(
#     X=X, X_test=X_test, y=y, params=params, folds=folds, model_type="lgb",
#     eval_metric="group_mae", plot_feature_importance=True,
#     verbose=500, early_stopping_rounds=200, n_estimators=1500)
#
# sub["scalar_coupling_constant"] = result_dict_lgb["prediction"]
# sub.to_csv("submission.csv", index=False)

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
        verbose=500, early_stopping_rounds=200, n_estimators=3000)
    X_short.loc[X_short["type"] == t, "oof"] = result_dict_lgb["oof"]
    X_short_test.loc[X_short_test["type"] == t, "prediction"] = \
        result_dict_lgb["prediction"]

sub["scalar_coupling_constant"] = X_short_test["prediction"]
sub.to_csv("submission_t.csv", index=False)
sub.head()
