import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


class ml_models():
    """
    A class that defines 1D Convolutional Neural Network models for learning PDE dynamics,
    specifically designed to map u(x, t) → u(x, t+Δt) for time-evolving spatial systems.

    Attributes:
        nx (int): Number of spatial grid points in the input profile.
    """
    def __init__(self, nx, *args, **kwargs):
        """
        Initializes the model class with grid resolution.

        Args:
            nx (int): Number of spatial grid points in the input data.
        """
        self.nx = nx
        
    def cnn_model_1d(self):
        """
        Builds a 1D CNN model that maps an input field u(x)^n to the next time step u(x)^{n+1}.
        The model preserves spatial structure using only convolutional layers.

        Returns:
            model_cnn (tf.keras.Model): Compiled 1D CNN model ready for training.
        """
        model_cnn = models.Sequential(
            [
                # Input shape: (nx, 1) — scalar value at each grid point
                tf.keras.Input(shape=(self.nx,1)),                
                # First convolutional layer with 96 filters
                layers.Conv1D(96, kernel_size=3, padding='same', activation='relu',name='conv_l1'),
                # Second convolutional layer with 64 filters
                layers.Conv1D(64, kernel_size=3, padding='same', activation='relu',name='conv_l2'),
                # Output layer: 1 filter per grid point (predicting u^{n+1}(x))
                layers.Conv1D(1, kernel_size=1, padding='same', name='output'),

            ]
        )
        return model_cnn

class data_generator():
    
    def __init__(self, path, dim, csv=1, npz=0, x_res=100, *args, **kwargs):
        
        self.filename = path
        self.dim = dim
        self.csv = csv
        self.npz = npz
        self.nx = x_res

    def read_1d(self):
        
        if (self.dim != 1):
            raise ValueError ("1D read function called for Dimensions > 1")
        
        if (self.csv):
            files = sorted([f for f in os.listdir(self.filename) if f.startswith('data_') and f.endswith('.csv')])
            x_shape = self.nx
            nt = len(files)
            
            var_u_all = np.zeros((nt-1, x_shape+1, 1))
            for i, file in enumerate(files[:-1]):
                data = pd.read_csv(os.path.join(self.filename, file))
                var_u = data.filter(like='u_t').values.flatten()
                var_x = data["x"]
                var_t = data["t"].iloc[0]
                var_u_all[i, 0, 0] = var_t
                var_u_all[i, 1:, 0] = var_u
        
        elif (self.npz):
            data = np.load(self.filename)
            var_u = data['u'][:-1]
            var_t = data['time'][:-1]
            nt = len(var_t)-1
            x_shape = self.nx
            var_u_all = np.zeros((nt, x_shape+1, 1))
            
            var_u_all[:, 0, 0] = var_t
            var_u_all[:, 1:, 0] = var_u
        
        else:
            raise ValueError ("No datafile format specified!")
            
            
        X = var_u_all[:-1, 1:, 0]
        Y = var_u_all[1:, 1:, 0]
        
        column_names = [f"u{i}" for i  in range(self.nx)]
        X_df = pd.DataFrame(X, columns=column_names)
        Y_df = pd.DataFrame(Y, columns=column_names)
        
        return X_df, Y_df