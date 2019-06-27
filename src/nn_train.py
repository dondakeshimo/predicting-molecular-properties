from keras import backend as K
from keras import callbacks
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.models import Model
import numpy as np
import time


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
    print(model.summary)
    return model


def train_nn_model(X, X_test, y, folds, model,
                   loss="mae", verbose=1, epochs=100, batch_size=32):

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

        y_pred_valid = model.predict(X_valid).reshape(-1,)
        y_pred = model.predict(X_test).reshape(-1,)

        accuracy = np.mean(np.abs(y_valid - y_pred_valid))
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
