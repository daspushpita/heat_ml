import os,sys
import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf 
import matplotlib as mlt
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib import animation
sys.path.append("../src")  # adjust path as needed
from data_generator import pinn_gen_D
from model import gen_pinn_model
from tensorflow.keras import models

#Load the data and train the model
data_dir = '/Users/pushpita/Documents/ML Projects/Project3/data/csv_files/'
save_path = '/Users/pushpita/Documents/ML Projects/Project3/saved_models/pinn/'
D = np.array([0.01, 0.03, 0.07, 0.09, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0])

model_ini = gen_pinn_model(data_dir, 
                        resolution=100, 
                        N_IC=100, N_BC=100, 
                        N_CP=100, N_diff=len(D),
                        diffusion_coeff=D, 
                        epoch=3000)

mymodel = model_ini.build_pinnmodel()
model_ini.train_model()
model_ini.save_model(save_path)

