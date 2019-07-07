import numpy as np
from tqdm import tqdm


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    show_mem_usage(df)

    for col in tqdm(df.columns):
        col_type = df[col].dtypes
        if col_type not in numerics:
            continue

        c_min = df[col].min()
        c_max = df[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and \
               c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)

            elif c_min > np.iinfo(np.int16).min and \
                    c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)

            elif c_min > np.iinfo(np.int32).min and \
                    c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)

            elif c_min > np.iinfo(np.int64).min and \
                    c_max < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)

        else:
            c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
            if c_min > np.finfo(np.float16).min and \
               c_max < np.finfo(np.float16).max and \
               c_prec == np.finfo(np.float16).precision:
                df[col] = df[col].astype(np.float16)

            elif c_min > np.finfo(np.float32).min and \
                    c_max < np.finfo(np.float32).max and \
                    c_prec == np.finfo(np.float32).precision:
                df[col] = df[col].astype(np.float32)

            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print("Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)"
              .format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df


def show_mem_usage(df):
    mb = 1024**2
    print(" ------------------------------------ ")
    print("{: >10}".format('Memory'))
    print(" ------------------------------------ ")
    print("{: >10}".format(df.memory_usage().sum() / mb))
    print(" ------------------------------------ ")
