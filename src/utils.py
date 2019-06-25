import pandas as pd


def map_atom_info(df, structures, atom_idx):
    df = pd.merge(df, structures, how='left',
                  left_on=['molecule_name', f'atom_index_{atom_idx}'],
                  right_on=['molecule_name', 'atom_index'])

    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df


def create_features(df):
    df['molecule_couples'] = \
        df.groupby('molecule_name')['id'].transform('count')
    df['molecule_dist_mean'] = \
        df.groupby('molecule_name')['dist'].transform('mean')
    df['molecule_dist_min'] = \
        df.groupby('molecule_name')['dist'].transform('min')
    df['molecule_dist_max'] = \
        df.groupby('molecule_name')['dist'].transform('max')
    df['atom_0_couples_count'] = \
        df.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')
    df['atom_1_couples_count'] = \
        df.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')

    df[f'molecule_atom_index_0_x_1_std'] = \
        df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('std')
    df[f'molecule_atom_index_0_y_1_mean'] = \
        df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('mean')
    df[f'molecule_atom_index_0_y_1_mean_diff'] = \
        df[f'molecule_atom_index_0_y_1_mean'] - df['y_1']
    df[f'molecule_atom_index_0_y_1_mean_div'] = \
        df[f'molecule_atom_index_0_y_1_mean'] / df['y_1']
    df[f'molecule_atom_index_0_y_1_max'] = \
        df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('max')
    df[f'molecule_atom_index_0_y_1_max_diff'] = \
        df[f'molecule_atom_index_0_y_1_max'] - df['y_1']
    df[f'molecule_atom_index_0_y_1_std'] = \
        df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('std')
    df[f'molecule_atom_index_0_z_1_std'] = \
        df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('std')
    df[f'molecule_atom_index_0_dist_mean'] = \
        df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('mean')
    df[f'molecule_atom_index_0_dist_mean_diff'] = \
        df[f'molecule_atom_index_0_dist_mean'] - df['dist']
    df[f'molecule_atom_index_0_dist_mean_div'] = \
        df[f'molecule_atom_index_0_dist_mean'] / df['dist']
    df[f'molecule_atom_index_0_dist_max'] = \
        df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('max')
    df[f'molecule_atom_index_0_dist_max_diff'] = \
        df[f'molecule_atom_index_0_dist_max'] - df['dist']
    df[f'molecule_atom_index_0_dist_max_div'] = \
        df[f'molecule_atom_index_0_dist_max'] / df['dist']
    df[f'molecule_atom_index_0_dist_min'] = \
        df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('min')
    df[f'molecule_atom_index_0_dist_min_diff'] = \
        df[f'molecule_atom_index_0_dist_min'] - df['dist']
    df[f'molecule_atom_index_0_dist_min_div'] = \
        df[f'molecule_atom_index_0_dist_min'] / df['dist']
    df[f'molecule_atom_index_0_dist_std'] = \
        df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('std')
    df[f'molecule_atom_index_0_dist_std_diff'] = \
        df[f'molecule_atom_index_0_dist_std'] - df['dist']
    df[f'molecule_atom_index_0_dist_std_div'] = \
        df[f'molecule_atom_index_0_dist_std'] / df['dist']
    df[f'molecule_atom_index_1_dist_mean'] = \
        df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('mean')
    df[f'molecule_atom_index_1_dist_mean_diff'] = \
        df[f'molecule_atom_index_1_dist_mean'] - df['dist']
    df[f'molecule_atom_index_1_dist_mean_div'] = \
        df[f'molecule_atom_index_1_dist_mean'] / df['dist']
    df[f'molecule_atom_index_1_dist_max'] = \
        df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('max')
    df[f'molecule_atom_index_1_dist_max_diff'] = \
        df[f'molecule_atom_index_1_dist_max'] - df['dist']
    df[f'molecule_atom_index_1_dist_max_div'] = \
        df[f'molecule_atom_index_1_dist_max'] / df['dist']
    df[f'molecule_atom_index_1_dist_min'] = \
        df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('min')
    df[f'molecule_atom_index_1_dist_min_diff'] = \
        df[f'molecule_atom_index_1_dist_min'] - df['dist']
    df[f'molecule_atom_index_1_dist_min_div'] = \
        df[f'molecule_atom_index_1_dist_min'] / df['dist']
    df[f'molecule_atom_index_1_dist_std'] = \
        df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('std')
    df[f'molecule_atom_index_1_dist_std_diff'] = \
        df[f'molecule_atom_index_1_dist_std'] - df['dist']
    df[f'molecule_atom_index_1_dist_std_div'] = \
        df[f'molecule_atom_index_1_dist_std'] / df['dist']
    df[f'molecule_atom_1_dist_mean'] = \
        df.groupby(['molecule_name', 'atom_1'])['dist'].transform('mean')
    df[f'molecule_atom_1_dist_min'] = \
        df.groupby(['molecule_name', 'atom_1'])['dist'].transform('min')
    df[f'molecule_atom_1_dist_min_diff'] = \
        df[f'molecule_atom_1_dist_min'] - df['dist']
    df[f'molecule_atom_1_dist_min_div'] = \
        df[f'molecule_atom_1_dist_min'] / df['dist']
    df[f'molecule_atom_1_dist_std'] = \
        df.groupby(['molecule_name', 'atom_1'])['dist'].transform('std')
    df[f'molecule_atom_1_dist_std_diff'] = \
        df[f'molecule_atom_1_dist_std'] - df['dist']
    df[f'molecule_type_0_dist_std'] = \
        df.groupby(['molecule_name', 'type_0'])['dist'].transform('std')
    df[f'molecule_type_0_dist_std_diff'] = \
        df[f'molecule_type_0_dist_std'] - df['dist']
    df[f'molecule_type_dist_mean'] = \
        df.groupby(['molecule_name', 'type'])['dist'].transform('mean')
    df[f'molecule_type_dist_mean_diff'] = \
        df[f'molecule_type_dist_mean'] - df['dist']
    df[f'molecule_type_dist_mean_div'] = \
        df[f'molecule_type_dist_mean'] / df['dist']
    df[f'molecule_type_dist_max'] = \
        df.groupby(['molecule_name', 'type'])['dist'].transform('max')
    df[f'molecule_type_dist_min'] = \
        df.groupby(['molecule_name', 'type'])['dist'].transform('min')
    df[f'molecule_type_dist_std'] = \
        df.groupby(['molecule_name', 'type'])['dist'].transform('std')
    df[f'molecule_type_dist_std_diff'] = \
        df[f'molecule_type_dist_std'] - df['dist']

    return df


def get_good_columns():
    return [
        'molecule_atom_index_0_dist_min',
        'molecule_atom_index_0_dist_max',
        'molecule_atom_index_1_dist_min',
        'molecule_atom_index_0_dist_mean',
        'molecule_atom_index_0_dist_std',
        'dist',
        'molecule_atom_index_1_dist_std',
        'molecule_atom_index_1_dist_max',
        'molecule_atom_index_1_dist_mean',
        'molecule_atom_index_0_dist_max_diff',
        'molecule_atom_index_0_dist_max_div',
        'molecule_atom_index_0_dist_std_diff',
        'molecule_atom_index_0_dist_std_div',
        'atom_0_couples_count',
        'molecule_atom_index_0_dist_min_div',
        'molecule_atom_index_1_dist_std_diff',
        'molecule_atom_index_0_dist_mean_div',
        'atom_1_couples_count',
        'molecule_atom_index_0_dist_mean_diff',
        'molecule_couples',
        'atom_index_1',
        'molecule_dist_mean',
        'molecule_atom_index_1_dist_max_diff',
        'molecule_atom_index_0_y_1_std',
        'molecule_atom_index_1_dist_mean_diff',
        'molecule_atom_index_1_dist_std_div',
        'molecule_atom_index_1_dist_mean_div',
        'molecule_atom_index_1_dist_min_diff',
        'molecule_atom_index_1_dist_min_div',
        'molecule_atom_index_1_dist_max_div',
        'molecule_atom_index_0_z_1_std',
        'y_0',
        'molecule_type_dist_std_diff',
        'molecule_atom_1_dist_min_diff',
        'molecule_atom_index_0_x_1_std',
        'molecule_dist_min',
        'molecule_atom_index_0_dist_min_diff',
        'molecule_atom_index_0_y_1_mean_diff',
        'molecule_type_dist_min',
        'molecule_atom_1_dist_min_div',
        'atom_index_0',
        'molecule_dist_max',
        'molecule_atom_1_dist_std_diff',
        'molecule_type_dist_max',
        'molecule_atom_index_0_y_1_max_diff',
        'molecule_type_0_dist_std_diff',
        'molecule_type_dist_mean_diff',
        'molecule_atom_1_dist_mean',
        'molecule_atom_index_0_y_1_mean_div',
        'molecule_type_dist_mean_div',
        'type']
