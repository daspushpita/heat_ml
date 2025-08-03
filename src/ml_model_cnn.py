import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input, regularizers


class ml_models():
    """
    A class that defines 1D Convolutional Neural Network models for learning PDE dynamics,
    specifically designed to map u(x, t) → u(x, t+Δt) for time-evolving spatial systems.

    Attributes:
        nx (int): Number of spatial grid points in the input profile.
    """
    def __init__(self, nx, dim, *args, **kwargs):
        """
        Initializes the model class with grid resolution.

        Args:
            nx (int): Number of spatial grid points in the input data.
        """
        self.nx = nx
        self.dim = dim
        
    def cnn_model_1d_single_diff(self, K):
        """
        Builds a 1D CNN model that maps an input field u(x)^n to the next time step u(x)^{n+1}.
        The model preserves spatial structure using only convolutional layers.

        Returns:
            model_cnn (tf.keras.Model): Compiled 1D CNN model ready for training.
        """
        # Input shape: (nx, 1) — scalar value at each grid point
        inputs = Input(shape=(self.nx,1), name = 'input_layer')
        x = layers.Conv1D(6, kernel_size=3, padding='same', activation='relu', name='conv_l1')(inputs)
        x = layers.Conv1D(6, kernel_size=3, padding='same', activation='relu',name='conv_l2')(x)
        if K > 1:
            x = layers.Conv1D(6, kernel_size=3, padding='same', activation='relu',name='conv_l3')(x)
        output = layers.Conv1D(K, kernel_size=1, padding='same', name='output')(x)
        model_cnn = models.Model(inputs=inputs, outputs=output)
        return model_cnn
    
    def cnn_model_1d_multistep_diff(self, K):
        """
        Builds a 1D CNN model that maps an input field u(x)^n to multiple next time steps u(x)^{n+1}, u(x)^{n+2}, u(x)^{n+3}.
        The model preserves spatial structure using only convolutional layers.

        Returns:
            model_cnn (tf.keras.Model): Compiled 1D CNN model ready for training.
        """
        # Input shape: (nx, 1) — scalar value at each grid point
        inputs = Input(shape=(self.nx, self.dim), name = 'input_layer')
        x = layers.Conv1D(9, kernel_size=3, padding='same', activation='relu', name='conv_l1')(inputs)
        x = layers.Conv1D(9, kernel_size=3, padding='same', activation='relu',name='conv_l2')(x)
        if K > 1:
            x = layers.Conv1D(6, kernel_size=3, padding='same', activation='relu',name='conv_l3')(x)
            
        x = layers.TimeDistributed(layers.Dense(136, activation='relu'), name='neural')(x)
        output = layers.Conv1D(K, kernel_size=1, padding='same', name='output')(x)
        model_cnn = models.Model(inputs=inputs, outputs=output)
        return model_cnn

    def cnn_model_1d_gen_diff(self):
        """
        Builds a 1D CNN model that maps an input field u(x)^n to the next time step u(x)^{n+1}.
        The model preserves spatial structure using only convolutional layers.

        Returns:
            model_cnn (tf.keras.Model): Compiled 1D CNN model ready for training.
        """
        # Input shape: (nx, 2) — scalar value at each grid point
        inputs = Input(shape=(self.nx,10), name = 'input_layer')
        # Convolutional layers to extract spatial features
        x = layers.Conv1D(32, kernel_size=5, padding='same', activation='tanh', name='conv1')(inputs)
        x = layers.Conv1D(32, kernel_size=5, padding='same', activation='tanh', name='conv2')(x)
        x = layers.Conv1D(6, kernel_size=5, padding='same', activation='tanh', name='conv3')(x)

        # Optional normalization
        x = layers.BatchNormalization(name='batchnorm')(x)

        # Flatten to go from (nx, filters) → (nx * filters)
        x = layers.Flatten(name='flatten')(x)

        # Dense block for global interactions across space
        x = layers.Dense(128, activation='tanh', name='dense1')(x)
        x = layers.Dense(self.nx, activation='linear', name='dense2')(x)

        # Reshape back to (nx, 1) as final output
        output = layers.Reshape((self.nx, 1), name='output')(x)

        # Define the model
        model_cnn = models.Model(inputs=inputs, outputs=output, name='cnn_pos_enc_model')

        return model_cnn
    

class Positional_Encoding():
    
    def __init__(self, x, d_model, n_user=10000, *args, **kwargs):
        self.x = x
        self.d_model = d_model
        self.n_user = n_user
    
    def sinosoidal_encoding(self):
        nk = len(self.x)
        n_freq = self.d_model//2
        pe_matrix = np.zeros((nk, self.d_model))
        for i in range(n_freq):
            theta = self.x / (self.n_user ** (2 * i / self.d_model))
            pe_matrix[:, 2*i] = np.sin(theta)
            pe_matrix[:, 2*i+1] = np.cos(theta)
        return pe_matrix
    
class data_generator():
    """
    Reads 1D PDE solution snapshots from disk and returns X → Y training data,
    optionally skipping time steps and holding out test data.
    """
    def __init__(self, path, holdout, nsteps, dim, csv=1, npz=0, x_res=100, gen=0, skip=0, *args, **kwargs):
        
        self.filename = path
        self.dim = dim
        self.csv = csv
        self.npz = npz
        self.nx = x_res
        self.gen = gen
        self.holdout = holdout
        self.skip = skip
        self.nsteps = nsteps

    def read_1d(self):
        
        if (self.dim != 1):
            raise ValueError ("1D read function called for Dimensions > 1")
        
        if (self.csv):
            files = sorted([f for f in os.listdir(self.filename) if f.startswith('data_') and f.endswith('.csv')])
            files = files if self.skip==0 else files[::self.skip]
            x_shape = self.nx
            data = pd.read_csv(os.path.join(self.filename, files[0]))
            self.x = data["x"].values
            nt = len(files)
            var_u_all = np.zeros((nt, x_shape, 3))
            for i, file in enumerate(files):
                data = pd.read_csv(os.path.join(self.filename, file))
                var_u = data.filter(like='u_t').values.flatten()
                var_x = data["x"]
                var_t = data["t"].iloc[0]

                var_u_all[i, :, 0] = var_u
                var_u_all[i, :, 1] = var_t
                var_u_all[i, :, 2] = var_x
                                
        elif (self.npz):
            data = np.load(self.filename)
            var_u = data['u'][:-1]
            var_t = data['time'][:-1]
            var_x = data['x']
            nt = len(var_t)
            x_shape = self.nx
            var_u_all = np.zeros((nt, x_shape, 3))

            var_u_all[:, :, 0] = var_u
            var_u_all[:, :, 1] = var_t            
            var_u_all[:, :, 2] = var_x
        
        else:
            raise ValueError ("No datafile format specified!")
            
            
        X = var_u_all[:-(self.nsteps), :, :]  # shape: (nt - nsteps, nx)

        # Build multi-step Y
        Y = np.stack([var_u_all[i+1:i+1+self.nsteps, :, 0]  # shape: (nsteps, nx)
            for i in range(nt - self.nsteps)])
        Y = np.transpose(Y, (0, 2, 1))  # shape: (samples, nx, nsteps)
        
        # Split training vs held-out
        if self.holdout > 0:
            X_train = X[:-self.holdout]
            Y_train = Y[:-self.holdout]
            X_test = X[-self.holdout:]
            Y_test = Y[-self.holdout:]
        else:
            X_train, Y_train = X, Y
            X_test, Y_test = None, None  # or np.empty(...)
        
        # column_names = [f"u{i}" for i  in range(self.nx)]
        # X_train_df = pd.DataFrame(X_train, columns=column_names)
        # Y_train_df = pd.DataFrame(Y_train, columns=column_names)
        
        if self.gen:
            x_grid = self.x
            return X_train, Y_train, X_test, Y_test, x_grid
        else:
            return X_train, Y_train, X_test, Y_test
    
