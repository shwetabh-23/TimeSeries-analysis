from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

def lstm_model_univariate():
    model = Sequential()
    model.add(InputLayer((5, 1)))
    model.add(LSTM(64))
    model.add(Dense(8, 'relu'))
    model.add(Dense(1, 'linear'))
    print(model.summary())
    return model

def lstm_model_multivariate():
    model = Sequential()
    model.add(InputLayer((6, 6)))
    model.add(LSTM(64))
    model.add(Dense(8, 'relu'))
    model.add(Dense(2, 'linear'))
    print(model.summary())
    return model

def compile_and_fit(model, X_train, y_train, X_val, y_val, chkpt_dir, model_save_dir):
    save_model_path = ModelCheckpoint(chkpt_dir, save_best_only = True)
    model.compile(loss = MeanSquaredError(), optimizer = Adam(learning_rate = 0.0001), metrics = [RootMeanSquaredError()])

    model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 10, callbacks = [save_model_path])
    model.save(model_save_dir)

def predict_and_plot_univariate(model, X_test, y_actual, modelname):
    
    if modelname == 'Univariate':
        y_pred = model.predict(X_test).flatten()

        plt.figure(figsize=(8, 6)) 
        plt.plot(np.array(y_actual), label='Actual', marker='o')
        plt.plot(np.array(y_pred), label='Predicted', marker='x')
        plt.xlabel('Data Point')
        plt.ylabel('Values')
        plt.title('Actual vs. Predicted Comparison')
        plt.legend()
        plt.savefig('comparison_plot_univariate.png')  
    
def predict_and_plot_multivariate(model, X_test, y_temp, y_pressure, modelname):
        results = model.predict(X_test)
        temp_pred = [a[0] for a in results]
        pressure_pred = [a[1] for a in results]

        plt.figure(1, figsize=(8, 6)) 
        plt.plot(np.array(y_temp), label='Actual', marker='o')
        plt.plot(np.array(temp_pred), label='Predicted', marker='x')
        plt.xlabel('Data Point')
        plt.ylabel('Values')
        plt.title('Actual vs. Predicted Comparison')
        plt.legend()
        plt.savefig('comparison_plot_multivariate_temp.png')

        plt.figure(2, figsize=(8, 6)) 
        plt.plot(np.array(y_pressure), label='Actual', marker='o')
        plt.plot(np.array(pressure_pred), label='Predicted', marker='x')
        plt.xlabel('Data Point')
        plt.ylabel('Values')
        plt.title('Actual vs. Predicted Comparison')
        plt.legend()
        plt.savefig('comparison_plot_multivariate_pressure.png')
