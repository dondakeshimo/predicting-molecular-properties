
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
    df[f"molecule_atom_index_0_x_1_std"] = \
        df[f"molecule_atom_index_0_x_1_std"].fillna(0)
    df[f"molecule_atom_index_0_y_1_mean"] = \
        df.groupby(["molecule_name", "atom_index_0"])["y_1"].transform("mean")
    df[f"molecule_atom_index_0_y_1_mean_diff"] = \
        df[f"molecule_atom_index_0_y_1_mean"] - df["y_1"]
    df[f"molecule_atom_index_0_y_1_mean_div"] = \
        df[f"molecule_atom_index_0_y_1_mean"] / df["y_1"]
    df[f"molecule_atom_index_0_y_1_mean_div"] = \
        df[f"molecule_atom_index_0_y_1_mean"].fillna(
            df[f"molecule_atom_index_0_y_1_mean"].max() * 10)
    df[f"molecule_atom_index_0_y_1_max"] = \
        df.groupby(["molecule_name", "atom_index_0"])["y_1"].transform("max")
    df[f"molecule_atom_index_0_y_1_max_diff"] = \
        df[f"molecule_atom_index_0_y_1_max"] - df["y_1"]
    df[f"molecule_atom_index_0_y_1_std"] = \
        df.groupby(["molecule_name", "atom_index_0"])["y_1"].transform("std")
    df[f"molecule_atom_index_0_y_1_std"] = \
        df[f"molecule_atom_index_0_y_1_std"].fillna(0)
    df[f"molecule_atom_index_0_z_1_std"] = \
        df.groupby(["molecule_name", "atom_index_0"])["z_1"].transform("std")
    df[f"molecule_atom_index_0_z_1_std"] = \
        df[f"molecule_atom_index_0_z_1_std"].fillna(0)
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
    df[f"molecule_atom_index_0_dist_std"] = \
        df[f"molecule_atom_index_0_dist_std"].fillna(0)
    df[f"molecule_atom_index_0_dist_std_diff"] = \
        df[f"molecule_atom_index_0_dist_std"] - df["dist"]
    df[f"molecule_atom_index_0_dist_std_div"] = \
        df[f"molecule_atom_index_0_dist_std"] / df["dist"]
    df[f"molecule_atom_index_0_dist_std_div"] = \
        df[f"molecule_atom_index_0_dist_std_div"].fillna(
            df[f"molecule_atom_index_0_dist_std_div"].max() * 10)
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
    df[f"molecule_atom_index_1_dist_std"] = \
        df[f"molecule_atom_index_1_dist_std"].fillna(0)
    df[f"molecule_atom_index_1_dist_std_diff"] = \
        df[f"molecule_atom_index_1_dist_std"] - df["dist"]
    df[f"molecule_atom_index_1_dist_std_div"] = \
        df[f"molecule_atom_index_1_dist_std"] / df["dist"]
    df[f"molecule_atom_index_1_dist_std_div"] = \
        df[f"molecule_atom_index_1_dist_std_div"].fillna(
            df[f"molecule_atom_index_1_dist_std_div"].max() * 10)
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
    df[f"molecule_atom_1_dist_std"] = \
        df[f"molecule_atom_1_dist_std"].fillna(0)
    df[f"molecule_atom_1_dist_std_diff"] = \
        df[f"molecule_atom_1_dist_std"] - df["dist"]
    df[f"molecule_type_0_dist_std"] = \
        df.groupby(["molecule_name", "type_0"])["dist"].transform("std")
    df[f"molecule_type_0_dist_std"] = \
        df[f"molecule_type_0_dist_std"].fillna(0)
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
    df[f"molecule_type_dist_std"] = \
        df[f"molecule_type_dist_std"].fillna(0)
    df[f"molecule_type_dist_std_diff"] = \
        df[f"molecule_type_dist_std"] - df["dist"]

    return df


def get_good_columns():
    return [
        "bond_lengths_mean_1",
        "bond_lengths_std_1",
        "bond_lengths_std_0",
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
