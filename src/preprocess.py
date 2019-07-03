import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

import artgor_utils
import handle_files


def map_atom_info(df, structures, atom_idx):
    df = pd.merge(df, structures, how="left",
                  left_on=["molecule_name", f"atom_index_{atom_idx}"],
                  right_on=["molecule_name", "atom_index"])

    df = df.drop("atom_index", axis=1)

    df = df.rename(columns={
        "atom": f"atom_{atom_idx}",
        "x": f"x_{atom_idx}",
        "y": f"y_{atom_idx}",
        "z": f"z_{atom_idx}",
        "n_bonds": f"n_bonds_{atom_idx}",
        "bond_lengths_mean": f"bond_lengths_mean_{atom_idx}",
        "bond_lengths_std": f"bond_lengths_std_{atom_idx}"
    })

    return df


def calc_dist(df):
    df_p_0 = df[["x_0", "y_0", "z_0"]].values
    df_p_1 = df[["x_1", "y_1", "z_1"]].values

    df["dist"] = np.linalg.norm(df_p_0 - df_p_1, axis=1)
    df["dist_x"] = (df["x_0"] - df["x_1"]) ** 2
    df["dist_y"] = (df["y_0"] - df["y_1"]) ** 2
    df["dist_z"] = (df["z_0"] - df["z_1"]) ** 2

    return df


def create_features_full(df):
    df["molecule_couples"] = \
        df.groupby("molecule_name")["id"].transform("count")
    df["molecule_dist_mean"] = \
        df.groupby("molecule_name")["dist"].transform("mean")
    df["molecule_dist_min"] = \
        df.groupby("molecule_name")["dist"].transform("min")
    df["molecule_dist_max"] = \
        df.groupby("molecule_name")["dist"].transform("max")
    df["atom_0_couples_count"] = \
        df.groupby(["molecule_name", "atom_index_0"])["id"].transform("count")
    df["atom_1_couples_count"] = \
        df.groupby(["molecule_name", "atom_index_1"])["id"].transform("count")

    num_cols = ["x_1", "y_1", "z_1", "dist", "dist_x", "dist_y", "dist_z"]
    cat_cols = ["atom_index_0", "atom_index_1", "type", "atom_1", "type_0"]
    aggs = ["mean", "max", "std", "min"]
    for col in cat_cols:
        df[f"molecule__{col}__count"] = \
            df.groupby("molecule_name")[col].transform("count")

    for cat_col in tqdm(cat_cols):
        for num_col in tqdm(num_cols):
            for agg in aggs:
                col = f"molecule__{cat_col}__{num_col}__{agg}"
                df[col] = df.groupby(["molecule_name", cat_col])[num_col] \
                            .transform(agg)
                if agg == "std":
                    df[col] = df[col].fillna(0)

                df[col + "__diff"] = df[col] - df[num_col]

                df[col + "__div"] = df[col] / df[num_col]
                df[col + "__div"] = df[col] / df[num_col].replace(0, 1e-10)

    return df


def create_basic_features(df):
    df["molecule_couples"] = \
        df.groupby("molecule_name")["id"].transform("count")
    df["molecule_dist_mean"] = \
        df.groupby("molecule_name")["dist"].transform("mean")
    df["molecule_dist_min"] = \
        df.groupby("molecule_name")["dist"].transform("min")
    df["molecule_dist_max"] = \
        df.groupby("molecule_name")["dist"].transform("max")
    df["atom_0_couples_count"] = \
        df.groupby(["molecule_name", "atom_index_0"])["id"].transform("count")
    df["atom_1_couples_count"] = \
        df.groupby(["molecule_name", "atom_index_1"])["id"].transform("count")

    return df


def create_extra_features(df, good_columns):
    columns = [g.split("__") for g in good_columns]
    columns = sorted(columns, key=lambda x: len(x))
    for cols in tqdm(columns):
        if len(cols) == 1:
            continue
        elif len(cols) == 3:
            _, col, _ = cols
            df[f"molecule__{col}__count"] = \
                df.groupby("molecule_name")[col].transform("count")
        elif len(cols) == 4:
            _, cat, num, agg = cols
            col = f"molecule__{cat}__{num}__{agg}"
            df[col] = df.groupby(["molecule_name", cat])[num] \
                        .transform(agg)
            if agg == "std":
                df[col] = df[col].fillna(0)
        elif len(cols) == 5:
            _, cat, num, agg, cal = cols
            col = f"molecule__{cat}__{num}__{agg}"
            if col not in df.columns:
                df[col] = df.groupby(["molecule_name", cat])[num] \
                            .transform(agg)
                if agg == "std":
                    df[col] = df[col].fillna(0)

            if cal == "diff":
                df[col + "__diff"] = df[col] - df[num]

            if cal == "div":
                df[col + "__div"] = df[col] / df[num].replace(0, 1e-10)

    return df


def get_good_columns(file_folder="../data", col_num=50):
    importance = pd.read_csv(f"{file_folder}/feature_importance.csv")
    importance = \
        importance.groupby(["feature"]).mean() \
        .sort_values(by=["importance"], ascending=False)
    good_columns = list(importance.index.values)[:col_num]
    good_columns.append("type")
    return good_columns


def get_atom_rad_en(structures):
    atomic_radius = {"H": 0.38, "C": 0.77, "N": 0.75, "O": 0.73, "F": 0.71}

    fudge_factor = 0.05
    atomic_radius = {k: v + fudge_factor for k, v in atomic_radius.items()}

    electronegativity = {"H": 2.2, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98}

    atoms = structures["atom"].values
    atoms_en = [electronegativity[x] for x in atoms]
    atoms_rad = [atomic_radius[x] for x in atoms]

    structures["EN"] = atoms_en
    structures["rad"] = atoms_rad

    return structures


def calc_bonds(structures):
    i_atom = structures["atom_index"].values
    p = structures[["x", "y", "z"]].values
    p_compare = p
    m = structures["molecule_name"].values
    m_compare = m
    r = structures["rad"].values
    r_compare = r

    source_row = np.arange(len(structures))
    max_atoms = 28

    bonds = np.zeros((len(structures) + 1, max_atoms + 1),
                     dtype=np.int8)
    bond_dists = np.zeros((len(structures) + 1, max_atoms + 1),
                          dtype=np.float32)

    print("Calculating bonds")

    for i in tqdm(range(max_atoms - 1)):
        p_compare = np.roll(p_compare, -1, axis=0)
        m_compare = np.roll(m_compare, -1, axis=0)
        r_compare = np.roll(r_compare, -1, axis=0)

        # Are we still comparing atoms in the same molecule?
        mask = np.where(m == m_compare, 1, 0)

        dists = np.linalg.norm(p - p_compare, axis=1) * mask
        r_bond = r + r_compare

        bond = np.where(np.logical_and(dists > 0.0001, dists < r_bond), 1, 0)

        source_row = source_row
        target_row = source_row + i + 1
        target_row = np.where(
            np.logical_or(target_row > len(structures), mask == 0),
            len(structures), target_row)

        source_atom = i_atom
        target_atom = i_atom + i + 1
        target_atom = np.where(
            np.logical_or(target_atom > max_atoms, mask == 0),
            max_atoms, target_atom)

        bonds[(source_row, target_atom)] = bond
        bonds[(target_row, source_atom)] = bond
        bond_dists[(source_row, target_atom)] = dists
        bond_dists[(target_row, source_atom)] = dists

    bonds = np.delete(bonds, axis=0, obj=-1)
    bonds = np.delete(bonds, axis=1, obj=-1)
    bond_dists = np.delete(bond_dists, axis=0, obj=-1)
    bond_dists = np.delete(bond_dists, axis=1, obj=-1)

    print("Counting and condensing bonds")

    bonds_numeric = [
        [i for i, x in enumerate(row) if x]
        for row in tqdm(bonds)
    ]
    bond_lengths = [
        [dist for i, dist in enumerate(row) if i in bonds_numeric[j]]
        for j, row in enumerate(tqdm(bond_dists))
    ]

    bond_lengths_mean = [np.mean(x) for x in bond_lengths]
    bond_lengths_std = [np.std(x) for x in bond_lengths]
    n_bonds = [len(x) for x in bonds_numeric]

    bond_data = {"n_bonds": n_bonds,
                 "bond_lengths_mean": bond_lengths_mean,
                 "bond_lengths_std": bond_lengths_std}
    bond_df = pd.DataFrame(bond_data)
    structures = structures.join(bond_df)

    return structures


def encode_str(train, test, good_columns):
    for f in ["atom_0", "atom_1", "type_0", "type"]:
        if f in good_columns:
            lbl = LabelEncoder()
            lbl.fit(list(train[f].values) + list(test[f].values))
            train[f] = lbl.transform(list(train[f].values))
            test[f] = lbl.transform(list(test[f].values))

    return train, test


def main():
    file_folder = "../data"
    train, test, structures, contrib = \
        handle_files.load_data_from_csv(file_folder)

    train = pd.merge(train, contrib, how="left",
                     left_on=["molecule_name", "atom_index_0",
                              "atom_index_1", "type"],
                     right_on=["molecule_name", "atom_index_0",
                               "atom_index_1", "type"])

    structures = get_atom_rad_en(structures)
    structures = calc_bonds(structures)

    train = train.iloc[:100000]
    test = test.iloc[:100000]

    train = map_atom_info(train, structures, 0)
    train = map_atom_info(train, structures, 1)
    test = map_atom_info(test, structures, 0)
    test = map_atom_info(test, structures, 1)

    train = calc_dist(train)
    test = calc_dist(test)

    train["type_0"] = train["type"].apply(lambda x: x[0])
    test["type_0"] = test["type"].apply(lambda x: x[0])

    train = create_features_full(train)
    test = create_features_full(test)

    for f in tqdm(["atom_0", "atom_1", "type_0", "type"]):
        lbl = LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

    X = train.drop(
        ["id", "scalar_coupling_constant", "molecule_name",
         "fc", "dso", "sd", "pso"], axis=1)
    y = train["scalar_coupling_constant"]
    X_test = test.drop(["id", "molecule_name"], axis=1)

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

    result_dict_lgb = artgor_utils.train_model_regression(
        X=X, X_test=X_test, y=y, params=params, folds=folds,
        model_type="lgb", eval_metric="group_mae",
        plot_feature_importance=True,
        verbose=300, early_stopping_rounds=1000, n_estimators=3000)

    result_dict_lgb["feature_importance"].to_csv(
        f"{file_folder}/feature_importance.csv", index=False)


if __name__ == "__main__":
    main()
