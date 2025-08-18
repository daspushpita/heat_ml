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


# === Paths ===
data_dir = '/Users/pushpita/Documents/ML Projects/Project3/data/csv_files/'
data_path = os.path.join(data_dir, 'diffusion0.05')
model_path = '/Users/pushpita/Documents/ML Projects/Project3/saved_models/pinn/pinn.weights.h5'
png_dir = "/Users/pushpita/Documents/ML Projects/Project3/plot/animations/PINN/gen_diffusion"
os.makedirs(png_dir, exist_ok=True)

# === Trained diffusion coefficients ===
D_train = np.array([0.01, 0.03, 0.07, 0.09, 0.1, 0.2])
D_test = 0.08  # Diffusion value to test

# === Utility: Get time array and x grid ===
def get_time_and_x(data_path):
    files = sorted(f for f in os.listdir(data_path) if f.startswith('data_') and f.endswith('.csv'))
    nt = len(files) - 1
    time_array = np.zeros(nt)

    data_sample = pd.read_csv(os.path.join(data_path, files[1]))
    x = data_sample["x"].values

    for i, file in enumerate(files[1:]):
        df = pd.read_csv(os.path.join(data_path, file))
        time_array[i] = df["t"].iloc[0]
    
    return time_array, x

# === Utility: Load true solutions ===
def load_true_solutions(data_path):
    files = sorted(f for f in os.listdir(data_path) if f.startswith('data_') and f.endswith('.csv'))
    nt = len(files) - 1
    time_array = np.zeros(nt)
    solutions = []

    for i, file in enumerate(files[1:]):
        df = pd.read_csv(os.path.join(data_path, file))
        time_array[i] = df["t"].iloc[0]
        u_true = df.filter(like='u_t').values.flatten()
        solutions.append(u_true)

    return np.array(solutions), time_array

# === Load model and predict ===
model = gen_pinn_model(data_dir, resolution=100, N_IC=100, N_BC=100, N_CP=100,
                       N_diff=len(D_train), diffusion_coeff=D_train, epoch=2000)
model.build_pinnmodel()
model.load_model(model_path)

t_vals, x_vals = get_time_and_x(data_path)
u_preds = model.predict(x_vals, t_vals, D=D_test).numpy()

# === Load true data for comparison (optional) ===
u_true_all, time_array = load_true_solutions(data_path)

# === Build predictions for all timesteps ===
u_preds_all = []
for i, t_val in enumerate(t_vals):
    t_input = np.full_like(x_vals, t_val)  # repeat same t for each x
    u_pred = model.predict(x_vals, t_input, D=D_test).numpy().flatten()
    u_preds_all.append(u_pred)

u_preds_all = np.array(u_preds_all)  # shape (nt, nx)
u_true_all = np.array(u_true_all)    # already loaded above

# === Compute relative L2 error ===
l2_error = np.linalg.norm(u_preds_all - u_true_all) / np.linalg.norm(u_true_all)
print(f"Relative L2 error: {l2_error:.6f}")

# === Setup figure ===
fig, ax = plt.subplots(figsize=(6, 4))
line_pred, = ax.plot([], [], lw=2, color='tab:red', label='Predicted')
line_true, = ax.plot([], [], lw=2, color='tab:green', linestyle='--', label='True')
ax.set_xlabel("x")
ax.set_ylabel("u(x, t)")
title = ax.set_title("")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()

# === Frame update function ===
def update_frame(i):
    t_val = time_array[i]
    t_input = np.full_like(x_vals, t_val)
    u_pred = model.predict(x_vals, t_input, D=D_test).numpy().flatten()
    u_true = u_true_all[i]
    line_pred.set_data(x_vals, u_pred)
    line_true.set_data(x_vals, u_true)
    title.set_text(f"t = {t_val:.3f}")

# === Generate and save frames ===
for i in range(len(t_vals)):
    update_frame(i)
    plt.savefig(os.path.join(png_dir, f"frame_{i:04d}.png"), dpi=110, bbox_inches="tight")
    
