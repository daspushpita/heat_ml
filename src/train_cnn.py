import os,sys
import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf 
import matplotlib.pyplot as plt

sys.path.append("../src")  # adjust path as needed
from data_generator import data_generator
from model import cnn_models
from tensorflow.keras import models
from sklearn.model_selection import train_test_split

#Load the data and train the model
output_dir = '/Users/pushpita/Documents/ML Projects/Project3/data/csv_files/diffusion0.1/'

holdout = 10
nsteps = 3

data = data_generator(output_dir, holdout=holdout, nsteps = nsteps, dim=1, csv=1, x_res=100)
X_train_base, Y_train_base, X_test, Y_test = data.read_1d()
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train_base[:,:,0:2], Y_train_base, test_size=0.2, 
                                                    shuffle=False)

model_instantiate = ml_models(nx = 100, dim=2)
my_model = model_instantiate.cnn_model_1d_multistep_diff(K=nsteps)

# Set learning rate here
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
my_model.build(input_shape=(None, 100, 1))  # assuming nx = 100

my_model.compile(loss='mse', optimizer=optimizer)
history = my_model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_validation, Y_validation))