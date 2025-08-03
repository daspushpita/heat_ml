import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input, regularizers, Model


class pinn_models():
    """
    A class that defines  Physics-informed Neural Network models for learning PDE dynamics.

    Attributes:
        nx (int): Number of spatial grid points in the input profile.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the model class with grid resolution.

        Args:
        """
        
    def build_pinnmodel(self, n_hidden=4, n_neurons=64):
        
        inputs = Input(shape=(2,), name = 'input_layer')
        x = inputs
        for i in range(n_hidden):
            x = layers.Dense(n_neurons, activation='tanh', name=f'dense_{i+1}')(x)
        
        output = layers.Dense(1, activation=None, name='output')(x)
        mymodel = Model(inputs=inputs, outputs=output, name='PINN')
        return mymodel
    
    # def pde_residual(self):

    

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
    
    
