# -*- encoding: utf-8 -*-

"""
Dynamically Train any AI-ML Model

Various types of models can be trained using this file, just define
the model parameters and definations, and then the model is trained
with the given parameters. The model, and performance metric is saved
on pre-specified directory which can be later loaded for evaluation.

The "DYNAMIC" nature of the code is limited to "global arguments"
defined in the `__main__` section, however other parameters like
model layers and parameters can be configured.

TODO setup a portfolio design for hyperparameter tuning of the model.
! Currently, all the models are defined under `./src/models/*.py`
! images and savedmodels are respectively available under `./output`

@author: Debmalya Pramanik
"""

import os   # miscellaneous os interfaces
import sys  # configuring python runtime environment

import numpy as np
import tensorflow as tf

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid');
plt.style.use('default-style');

### --- append additional paths to python runtime environment --- ###
# append `src` and sub-modules to call additional files these directory are
# project specific and not to be added under environment or $PATH variable
sys.path.append(os.path.join(".", "src", "models")) # derivative engines for model control
from lstm import BareLSTM

if __name__ == "__main__":
    _, xy_train_file_ = sys.argv # ! always get from `processed`
    DATA_DIR = os.path.join(".", "data", "processed")
    
    # ! define model save path directory & image save path
    IMAGES_DIR = os.path.join(".", "output", "images")
    MODELS_DIR = os.path.join(".", "output", "savedmodels")

    # ! hard code `n_lookback` and `n_forecast`
    n_lookback = 3 * 96 # 🧪 look into 7 days past records
    n_forecast = 2 * 96 # on t(-1) we want prediction for t(+1)

    # * read data from pre-processed file/generate data from source
    with open(os.path.join(DATA_DIR, xy_train_file_), "rb") as fileObj:
        x_train = np.load(fileObj)
        y_train = np.load(fileObj)

    ### *** Define LSTM Model Parameters *** ###
    # neural network parameters, parametric as much possible
    ACTIVATION_FUNCTION = "relu"

    # model tuning parameters
    LR_START = 1e-3
    LR_FINAL = 2e-4
    NUM_EPOCHS = 2
    BATCH_SIZE = 1024

    # callback and model monitoring criteria
    LR_FUNC = tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", factor = 0.2, patience = 5, min_lr = LR_FINAL)
    ES_FUNC = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 12, min_delta = 0.001, restore_best_weights = True)
    TM_FUNC = tf.keras.callbacks.TerminateOnNaN()

    # define the callbacks for model
    callbacks = [
        LR_FUNC, # learning rate
        ES_FUNC, # early stopping of model training
        TM_FUNC  # terminate model training on null value
    ]

    # define model using defined `agents`
    model = BareLSTM(
        input_shape = (n_lookback, 1), # ⚠ for univariate shape is always `(-1, 1)`
        output_shape = n_forecast, # 💿 high network will throw resource error
        activation = ACTIVATION_FUNCTION,
        model_name = "StackedLSTM-TestSequence"
    ).get_2_layer_lstm(units = [64, 32])
    print(model.summary(line_length = 127))
    
    # compile and train the model
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = LR_START, amsgrad = True),
        loss = tf.keras.losses.MeanSquaredError(name = "loss"),
        metrics = [tf.keras.metrics.RootMeanSquaredError(name = "RMSE"), tf.keras.metrics.MeanAbsoluteError(name = "MAE")]
    )

    # get the history from `.train` method
    history = model.fit(x_train, y_train, epochs = NUM_EPOCHS, batch_size = BATCH_SIZE, validation_split = 0.05, shuffle = True)

    # save the model as `h5` file
    model.save(os.path.join(MODELS_DIR, f"{model.name}.h5"))

    # plot and save the training metrics
    plt.figure(figsize = (27, 3))

    plt.subplot(121)
    plt.plot(history.history["loss"], label = "loss")
    plt.plot(history.history["val_loss"], label = "val_loss")

    plt.title("Loss Metric (AUC)")
    plt.legend()

    plt.subplot(122)
    plt.plot(history.history["MAE"], label = "MAE")
    plt.plot(history.history["val_MAE"], label = "val_MAE")

    plt.title("Mean Absolute Error (MAE)")
    plt.legend()
    plt.savefig(os.path.join(IMAGES_DIR, f"{model.name}.png"))
    plt.cla(); plt.clf() # close and clear axis, figure
