from keras import backend as K
from keras import callbacks
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.models import Model
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


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


def create_features(df):
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

    df[f"molecule_atom_index_0_x_1_std"] = \
        df.groupby(["molecule_name", "atom_index_0"])["x_1"].transform("std")
    df[f"molecule_atom_index_0_y_1_mean"] = \
        df.groupby(["molecule_name", "atom_index_0"])["y_1"].transform("mean")
    df[f"molecule_atom_index_0_y_1_mean_diff"] = \
        df[f"molecule_atom_index_0_y_1_mean"] - df["y_1"]
    df[f"molecule_atom_index_0_y_1_mean_div"] = \
        df[f"molecule_atom_index_0_y_1_mean"] / df["y_1"]
    df[f"molecule_atom_index_0_y_1_max"] = \
        df.groupby(["molecule_name", "atom_index_0"])["y_1"].transform("max")
    df[f"molecule_atom_index_0_y_1_max_diff"] = \
        df[f"molecule_atom_index_0_y_1_max"] - df["y_1"]
    df[f"molecule_atom_index_0_y_1_std"] = \
        df.groupby(["molecule_name", "atom_index_0"])["y_1"].transform("std")
    df[f"molecule_atom_index_0_z_1_std"] = \
        df.groupby(["molecule_name", "atom_index_0"])["z_1"].transform("std")
    df[f"molecule_atom_index_0_dist_mean"] = \
        df.groupby(["molecule_name", "atom_index_0"])["dist"].transform("mean")
    df[f"molecule_atom_index_0_dist_mean_diff"] = \
        df[f"molecule_atom_index_0_dist_mean"] - df["dist"]
    df[f"molecule_atom_index_0_dist_mean_div"] = \
        df[f"molecule_atom_index_0_dist_mean"] / df["dist"]
    df[f"molecule_atom_index_0_dist_max"] = \
        df.groupby(["molecule_name", "atom_index_0"])["dist"].transform("max")
    df[f"molecule_atom_index_0_dist_max_diff"] = \
        df[f"molecule_atom_index_0_dist_max"] - df["dist"]
    df[f"molecule_atom_index_0_dist_max_div"] = \
        df[f"molecule_atom_index_0_dist_max"] / df["dist"]
    df[f"molecule_atom_index_0_dist_min"] = \
        df.groupby(["molecule_name", "atom_index_0"])["dist"].transform("min")
    df[f"molecule_atom_index_0_dist_min_diff"] = \
        df[f"molecule_atom_index_0_dist_min"] - df["dist"]
    df[f"molecule_atom_index_0_dist_min_div"] = \
        df[f"molecule_atom_index_0_dist_min"] / df["dist"]
    df[f"molecule_atom_index_0_dist_std"] = \
        df.groupby(["molecule_name", "atom_index_0"])["dist"].transform("std")
    df[f"molecule_atom_index_0_dist_std_diff"] = \
        df[f"molecule_atom_index_0_dist_std"] - df["dist"]
    df[f"molecule_atom_index_0_dist_std_div"] = \
        df[f"molecule_atom_index_0_dist_std"] / df["dist"]
    df[f"molecule_atom_index_1_dist_mean"] = \
        df.groupby(["molecule_name", "atom_index_1"])["dist"].transform("mean")
    df[f"molecule_atom_index_1_dist_mean_diff"] = \
        df[f"molecule_atom_index_1_dist_mean"] - df["dist"]
    df[f"molecule_atom_index_1_dist_mean_div"] = \
        df[f"molecule_atom_index_1_dist_mean"] / df["dist"]
    df[f"molecule_atom_index_1_dist_max"] = \
        df.groupby(["molecule_name", "atom_index_1"])["dist"].transform("max")
    df[f"molecule_atom_index_1_dist_max_diff"] = \
        df[f"molecule_atom_index_1_dist_max"] - df["dist"]
    df[f"molecule_atom_index_1_dist_max_div"] = \
        df[f"molecule_atom_index_1_dist_max"] / df["dist"]
    df[f"molecule_atom_index_1_dist_min"] = \
        df.groupby(["molecule_name", "atom_index_1"])["dist"].transform("min")
    df[f"molecule_atom_index_1_dist_min_diff"] = \
        df[f"molecule_atom_index_1_dist_min"] - df["dist"]
    df[f"molecule_atom_index_1_dist_min_div"] = \
        df[f"molecule_atom_index_1_dist_min"] / df["dist"]
    df[f"molecule_atom_index_1_dist_std"] = \
        df.groupby(["molecule_name", "atom_index_1"])["dist"].transform("std")
    df[f"molecule_atom_index_1_dist_std_diff"] = \
        df[f"molecule_atom_index_1_dist_std"] - df["dist"]
    df[f"molecule_atom_index_1_dist_std_div"] = \
        df[f"molecule_atom_index_1_dist_std"] / df["dist"]
    df[f"molecule_atom_1_dist_mean"] = \
        df.groupby(["molecule_name", "atom_1"])["dist"].transform("mean")
    df[f"molecule_atom_1_dist_min"] = \
        df.groupby(["molecule_name", "atom_1"])["dist"].transform("min")
    df[f"molecule_atom_1_dist_min_diff"] = \
        df[f"molecule_atom_1_dist_min"] - df["dist"]
    df[f"molecule_atom_1_dist_min_div"] = \
        df[f"molecule_atom_1_dist_min"] / df["dist"]
    df[f"molecule_atom_1_dist_std"] = \
        df.groupby(["molecule_name", "atom_1"])["dist"].transform("std")
    df[f"molecule_atom_1_dist_std_diff"] = \
        df[f"molecule_atom_1_dist_std"] - df["dist"]
    df[f"molecule_type_0_dist_std"] = \
        df.groupby(["molecule_name", "type_0"])["dist"].transform("std")
    df[f"molecule_type_0_dist_std_diff"] = \
        df[f"molecule_type_0_dist_std"] - df["dist"]
    df[f"molecule_type_dist_mean"] = \
        df.groupby(["molecule_name", "type"])["dist"].transform("mean")
    df[f"molecule_type_dist_mean_diff"] = \
        df[f"molecule_type_dist_mean"] - df["dist"]
    df[f"molecule_type_dist_mean_div"] = \
        df[f"molecule_type_dist_mean"] / df["dist"]
    df[f"molecule_type_dist_max"] = \
        df.groupby(["molecule_name", "type"])["dist"].transform("max")
    df[f"molecule_type_dist_min"] = \
        df.groupby(["molecule_name", "type"])["dist"].transform("min")
    df[f"molecule_type_dist_std"] = \
        df.groupby(["molecule_name", "type"])["dist"].transform("std")
    df[f"molecule_type_dist_std_diff"] = \
        df[f"molecule_type_dist_std"] - df["dist"]

    return df


def get_good_columns():
    return [
        "bond_lengths_mean_1",
        "bond_lengths_std_0",
        "bond_lengths_std_1",
        "molecule_atom_index_0_dist_max",
        "bond_lengths_mean_0",
        "molecule_atom_index_0_dist_mean",
        "molecule_atom_index_0_dist_std",
        "molecule_couples",
        "molecule_atom_index_0_y_1_std",
        "molecule_dist_mean",
        "molecule_dist_max",
        "dist_y",
        "molecule_atom_index_0_z_1_std",
        "molecule_atom_index_1_dist_max",
        "molecule_atom_index_1_dist_min",
        "molecule_atom_index_0_x_1_std",
        "molecule_atom_index_1_dist_std",
        "molecule_atom_index_0_y_1_mean_div",
        "y_0",
        "molecule_atom_index_1_dist_mean",
        "molecule_atom_1_dist_mean",
        "x_0",
        "dist_x",
        "molecule_type_dist_std",
        "dist_z",
        "molecule_atom_index_1_dist_std_diff",
        "molecule_type_dist_mean_diff",
        "molecule_atom_index_0_dist_max_div",
        "molecule_atom_1_dist_std",
        "molecule_type_0_dist_std",
        "z_0",
        "molecule_type_dist_std_diff",
        "molecule_atom_index_0_y_1_mean_diff",
        "molecule_atom_index_0_dist_std_diff",
        "molecule_atom_index_0_dist_mean_div",
        "molecule_atom_index_0_dist_max_diff",
        "x_1",
        "molecule_type_dist_max",
        "molecule_atom_index_0_dist_std_div",
        "molecule_atom_index_0_dist_mean_diff",
        "molecule_atom_1_dist_std_diff",
        "molecule_atom_index_0_y_1_max_diff",
        "z_1",
        "molecule_atom_index_0_y_1_max",
        "molecule_atom_index_0_y_1_mean",
        "y_1",
        "molecule_type_0_dist_std_diff",
        "molecule_dist_min",
        "molecule_atom_index_1_dist_std_div",
        "molecule_atom_1_dist_min",
        "molecule_atom_index_1_dist_max_diff",
        "type"
    ]


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


def encode_str(train, test):
    good_columns = get_good_columns()
    for f in ["atom_0", "atom_1", "type_0", "type"]:
        if f in good_columns:
            lbl = LabelEncoder()
            lbl.fit(list(train[f].values) + list(test[f].values))
            train[f] = lbl.transform(list(train[f].values))
            test[f] = lbl.transform(list(test[f].values))

    return train, test


def create_nn_model(input_shape):
    input = Input(shape=(input_shape,))
    x = Dense(256, activation="relu", kernel_initializer="he_normal")(input)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation="relu", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation="relu", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    output = Dense(1, activation="linear")(x)
    model = Model(inputs=input, outputs=output)
    return model


def train_nn_model(X, X_test, y, params, folds, model,
                   loss="mae", verbose=1, epochs=100, batch_size=64):

    result_dict = {}

    # out-of-fold predictions on train data
    oof = np.zeros(len(X))

    # averaged predictions on train data
    prediction = np.zeros(len(X_test))

    # list of scores on folds
    scores = []

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f"Fold {fold_n + 1} started at {time.ctime()}")
        if type(X) == np.ndarray:
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = \
                X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        model.compile(loss=loss, optimizer="adam")
        es = callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0., patience=12,
            verbose=verbose, mode='min', baseline=None,
            restore_best_weights=True)
        rlr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.3, patience=4,
            min_lr=2e-6, mode='min', verbose=verbose)
        model.fit(
            X_train, y_train, validation_data=(X_valid, y_valid),
            callbacks=[es, rlr], epochs=epochs,
            batch_size=batch_size, verbose=verbose)

        y_pred_valid = model.predict(X_valid)
        y_pred = model.predict(X_test)

        accuracy = np.mean(np.abs(y_valid - y_pred_valid[0][:, 0]))
        scores.append(np.log(accuracy))

        oof[valid_index] = y_pred_valid.reshape(-1,)

        prediction += y_pred

        K.clear_session()

    prediction /= folds.n_splits

    print("CV mean score: {0:.4f}, std: {1:.4f}.".format(
        np.mean(scores), np.std(scores)))

    result_dict["oof"] = oof
    result_dict["prediction"] = prediction
    result_dict["scores"] = scores

    return result_dict
