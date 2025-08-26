import os,sys
import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf 
import joblib
from sklearn.base import clone
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from tensorflow.keras.callbacks import LambdaCallback

sys.path.append("../src")  # adjust path as needed
from data_generator import data_generator_simple_nn
from model import neural_net

output_dir = '/Users/pushpita/Documents/ML Projects/Project3/data/csv_files/diffusion0.1_highrestime/'

holdout = 100
nsteps = 1
x_resolution = 100
dim = 2  #choosing u and x 

data = data_generator_simple_nn(output_dir, holdout=holdout, 
                                nsteps = nsteps, dim=dim, 
                                csv=1, x_res=100)

grid_params = { 
    'model__n_layers': [2,3,4],
    'model__n_units': [24, 32, 64, 128],
    'model__activation' :['tanh', 'relu'],
    'model__learning_rate' : [1.e-4, 1.e-3, 1.e-2],
    'model__dropout_param': [0.0, 0.1, 0.5]
    }

k_fold = TimeSeriesSplit(n_splits = 5, test_size = 50)

def build_model(n_layers, n_units, activation, 
                learning_rate,dropout_param, 
                x_resolution=x_resolution, dim=dim, 
                nsteps=nsteps):
    
    model_instantiate = neural_net(n_layers=n_layers, 
                                n_units=n_units,
                                activation=activation,
                                nx = x_resolution, 
                                dim=dim,
                                K=nsteps,
                                reshape_output=False,
                                dropout=True,
                                dropout_param=dropout_param)
    
    my_model = model_instantiate.simple_nn_multistep()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    my_model.build(input_shape=(None, x_resolution, dim))  # assuming nx = 100
    my_model.compile(loss='mse', optimizer=optimizer)
    return my_model


print_callback = LambdaCallback(
    on_train_begin=lambda logs: print("Starting new model training..."),)

keras_reg = KerasRegressor(model=build_model,
                            model__x_resolution=x_resolution,
                            model__dim=dim,
                            model__nsteps=nsteps,
                            fit__epochs=200,
                            fit__batch_size=32,
                            fit__shuffle=False,
                            callbacks=[print_callback],
                            verbose=2)

grid = GridSearchCV(param_grid=grid_params, estimator=keras_reg,
                             scoring='neg_mean_squared_error',
                             n_jobs=4, cv=k_fold, return_train_score=True)

X_train_base, Y_train_base, X_test, Y_test = data.read_1d()

if nsteps > 1 or dim > 2:
    raise ValueError("This grid search is only for single step and 2D data (u and x).")
else:
    Y_train_base = Y_train_base.reshape(Y_train_base.shape[0], -1)
    Y_test = Y_test.reshape(Y_test.shape[0], -1)


best_params = grid.fit(X_train_base[:,:,:], Y_train_base)
y_pred = grid.best_estimator_.predict(X_test)

#Save the best parameters and model
joblib.dump(grid.best_params_, "best_params.pkl")
joblib.dump(grid.cv_results_, "cv_results.pkl")
grid.best_estimator_.model_.save_weights("best_model.weights.h5")