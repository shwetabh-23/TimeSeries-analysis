import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import model
import torch.optim as optim
import torch.nn as nn
import torch
from tensorflow_model import lstm_model_univariate, lstm_model_multivariate,compile_and_fit,  predict_and_plot
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from data import set_matrix_form_univariate, set_matrix_form_multivariate, create_multivariate_data

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

data = pd.read_csv(r'data\jena_climate_2009_2016.csv')
data.index = pd.to_datetime(data['Date Time'], format= '%d.%m.%Y %H:%M:%S')
#plt.plot(data['T (degC)'])
#plt.savefig('temp_var.jpg')

#breakpoint()
choice = input('Enter 1 for univariate analysis or 2 for multivariate analysis')
if choice == 1:
    model_name = 'Univariate'
    X, y = set_matrix_form_univariate(df = data['T (degC)'], window=5)
    X_train, y_train = X[:60000], y[:60000]
    X_val, y_val = X[60000:65000], y[60000:65000]
    X_test, y_test = X[65000:70000], y[65000:70000]

    chkpt_dir_univariate = r'save_model\univariate\lstm/'
    model_save_dir_multivariate = r'save_model\univariate\lstm/lstm_model.h5'

    if model_save_dir_multivariate:
        model = load_model(model_save_dir_multivariate)
    else:
        lstm_model = lstm_model_univariate()
        compile_and_fit(lstm_model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val
                        , chkpt_dir=chkpt_dir_univariate, model_save_dir=model_save_dir_multivariate)
        model = load_model(model_save_dir_multivariate)

    predict_and_plot(model, X_test, y_test, model_name)
else:
    model_name = 'Multivariate'
    multivariate_data = create_multivariate_data(data)
    X, y = set_matrix_form_multivariate(multivariate_data, 6)
    X_train, y_train = X[:60000], y[:60000]
    X_val, y_val = X[60000:65000], y[60000:65000]
    X_test, y_test = X[65000:70000], y[65000:70000]
    breakpoint()

    chkpt_dir_multivariate = r'save_model\multivariate\lstm/'
    model_save_dir_multivariate = r'save_model\multivariate\lstm/lstm_model.h5'

    if model_save_dir_multivariate and os.path.isfile(model_save_dir_multivariate):
        model = load_model(model_save_dir_multivariate)
    else:
        lstm_model = lstm_model_multivariate()
        compile_and_fit(lstm_model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val
                        , chkpt_dir=chkpt_dir_multivariate, model_save_dir=model_save_dir_multivariate)
        model = load_model(model_save_dir_multivariate)

    predict_and_plot(model, X_test, y_test, model_name)

    breakpoint()


