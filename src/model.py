import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input, regularizers
from data_generator import pinn_gen_D
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class cnn_models():
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
        x = layers.BatchNormalization(name='batchnorm')(x)
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(128, activation='tanh', name='dense1')(x)
        x = layers.Dense(self.nx, activation='linear', name='dense2')(x)
        output = layers.Reshape((self.nx, 1), name='output')(x)

        # Define the model
        model_cnn = models.Model(inputs=inputs, outputs=output, name='cnn_pos_enc_model')
        return model_cnn
    

class single_pinn_model():
    """
    A class that defines  Physics-informed Neural Network models for learning PDE dynamics.

    Attributes:
        nx (int): Number of spatial grid points in the input profile.
    """
    def __init__(self, data_path, resolution, 
                 N_IC, N_BC, N_CP, N_diff,
                 epoch=1000, diffusion_coeff=None,
                 learning_rate=1.e-3,
                 *args, **kwargs):
        """
        Initializes the model class with grid resolution.

        Args:
        """
        self.data_path = data_path
        self.resolution = resolution
        self.N_IC = N_IC
        self.N_BC = N_BC
        self.N_CP = N_CP
        self.N_diff = N_diff
        self.diffusion_coeff = diffusion_coeff
        self.epoch=epoch
        self.learning_rate= learning_rate
        np.random.seed(42)  # Set seed once
        
    def pinn_data_generator_single(self, IC, BC, CP):
        
        if IC==True:
            data = pd.read_csv(self.data_path+'data_0000.csv')
            x = data['x']
            u_xt = data['u_t0']
            idx = np.random.choice(len(x), size=self.N_IC, replace=False)
            x_ic = x[idx]
            t_ic = np.zeros_like(x_ic)
            u_xt_ic = u_xt[idx]
            x_tf_ic = tf.convert_to_tensor(x_ic, dtype=tf.float32)
            t_tf_ic = tf.convert_to_tensor(t_ic, dtype=tf.float32)
            u_tf_ic = tf.convert_to_tensor(u_xt_ic.to_numpy()[:, None], dtype=tf.float32)
            # X_input = tf.concat([x_tf_ic, t_tf_ic], axis=1)
            X_input = tf.concat([x_tf_ic[:, None], t_tf_ic[:, None]], axis=1)
            return X_input, u_tf_ic
        
        else:
            files = sorted([f for f in os.listdir(self.data_path) if f.startswith('data_') and f.endswith('.csv')])
            nt = len(files) - 1 #Ignoring the initial condition
            time_array = np.zeros((nt))
            for i, file in enumerate(files[1:]):
                data_all = pd.read_csv(os.path.join(self.data_path, file))
                var_t = data_all["t"].iloc[0]
                time_array[i] = var_t

            
            if BC == True:
                idt = np.random.choice(len(time_array), size=self.N_BC, replace=False)
                t_bc = time_array[idt]

                # Repeat x=0 and x=1 for each sampled time
                x_bc = np.tile([0.0, 1.0], self.N_BC)  # shape (2*N_BC,)
                t_bc = np.repeat(t_bc, 2)             # shape (2*N_BC,)
                u_bc = np.zeros_like(x_bc)            # (e.g., Dirichlet BC: u=0 at both ends)

                # Convert to tensors
                x_tf_bc = tf.convert_to_tensor(x_bc[:, None], dtype=tf.float32)
                t_tf_bc = tf.convert_to_tensor(t_bc[:, None], dtype=tf.float32)
                u_tf_bc = tf.convert_to_tensor(u_bc[:, None], dtype=tf.float32)
                X_input = tf.concat([x_tf_bc, t_tf_bc], axis=1)
                return X_input, u_tf_bc
            
            elif CP == True:
                x_cp = np.random.uniform(0.0, 1.0, size=self.N_CP)
                t_cp = np.random.uniform(np.min(time_array), np.max(time_array), size=self.N_CP)

                # Convert to tensors
                x_tf_cp = tf.convert_to_tensor(x_cp[:, None], dtype=tf.float32)
                t_tf_cp = tf.convert_to_tensor(t_cp[:, None], dtype=tf.float32)

                X_input = tf.concat([x_tf_cp, t_tf_cp], axis=1)
                return X_input
            
            else:
                raise ValueError("Train Data type not specified")

    def build_pinnmodel_single(self, n_hidden=4, n_neurons=50):
        
        inputs = Input(shape=(2,), name = 'input_layer')
        x = inputs
        for i in range(n_hidden):
            x = layers.Dense(n_neurons, activation='tanh', name=f'dense_{i+1}')(x)
        output = layers.Dense(1, activation=None, name='output')(x)
        mymodel = models.Model(inputs=inputs, outputs=output, name='PINN')
        self.model_cont = mymodel
        return mymodel
    
    def pde_residual_single(self, diff_const, X_tensor):

        x_tf = X_tensor[:, 0:1]
        t_tf = X_tensor[:, 1:2]
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x_tf, t_tf])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x_tf, t_tf])
                X_input = tf.concat([x_tf, t_tf], axis=1)  # shape (N, 2)
                u = self.model_cont(X_input)
            u_x = tape1.gradient(u, x_tf)
            u_t = tape1.gradient(u, t_tf)
                
        u_xx = tape2.gradient(u_x, x_tf)
        
        f = u_t - diff_const * u_xx
        return f
    
    def loss_terms_single(self, diff_const):
        
        X_ic_input, u_ic = self.pinn_data_generator_single(IC=True, BC=False, CP=False)
        u_ic_pred = self.model_cont (X_ic_input)
        loss_ic = tf.reduce_mean(tf.square(u_ic - u_ic_pred))
        
        X_bc_input, u_bc = self.pinn_data_generator_single(IC=False, BC=True, CP=False)
        u_bc_pred = self.model_cont (X_bc_input)
        loss_bc = tf.reduce_mean(tf.square(u_bc - u_bc_pred))
        
        X_cp_input = self.pinn_data_generator_single(IC=False, BC=False, CP=True)
        loss_pde = tf.reduce_mean(tf.square(self.pde_residual_single(diff_const, X_cp_input)))
        
        loss = loss_ic + loss_bc + loss_pde
        return loss, loss_ic, loss_bc, loss_pde
    
    def train_model_single(self, diff_const):
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        for epoch in range(self.epoch):
            with tf.GradientTape() as tape:
                loss, loss_ic, loss_bc, loss_pde = self.loss_terms_single(diff_const)
            
            grads = tape.gradient(loss, self.model_cont.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model_cont.trainable_variables))
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Total Loss = {loss.numpy():.4e} | IC = {loss_ic.numpy():.4e} | BC = {loss_bc.numpy():.4e} | PDE = {loss_pde.numpy():.4e}")
                
    def predict(self, x, t):        
        # Ensure x and t are column vectors
        x = np.reshape(x, (-1, 1))
        t = np.reshape(t, (-1, 1))
        
        # Convert to TensorFlow tensors
        x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
        t_tf = tf.convert_to_tensor(t, dtype=tf.float32)
        # Concatenate inputs
        X_input = tf.concat([x_tf, t_tf], axis=1)

        # Make prediction
        u_pred = self.model_cont(X_input)
        return u_pred
    
class gen_pinn_model():
    """
    A class that defines  Physics-informed Neural Network models for learning PDE dynamics.

    Attributes:
        nx (int): Number of spatial grid points in the input profile.
    """
    def __init__(self, data_path, resolution, 
                 N_IC, N_BC, N_CP, N_diff,
                 diffusion_coeff,
                 epoch=1000,
                 learning_rate=1.e-3,
                 print_loss=1,
                 *args, **kwargs):
        """
        Initializes the model class with grid resolution.

        Args:
        """
        self.data_path = data_path
        self.resolution = resolution
        self.N_IC = N_IC
        self.N_BC = N_BC
        self.N_CP = N_CP
        self.N_diff = N_diff
        self.diffusion_coeff = diffusion_coeff
        self.epoch=epoch
        self.learning_rate= learning_rate
        self.print_loss = print_loss

    ## All the functions for generalizing over the diffusion coefficients
    
    def build_pinnmodel(self, n_hidden=6, n_neurons=256):
        
        inputs = Input(shape=(3,), name = 'input_layer')
        x = inputs
        for i in range(n_hidden):
            x = layers.Dense(n_neurons, activation='tanh', name=f'dense_{i+1}')(x)
        output = layers.Dense(1, activation=None, name='output')(x)
        mymodel = models.Model(inputs=inputs, outputs=output, name='PINN')
        self.model_cont = mymodel
        return mymodel
    
    def pde_residual(self, X_tensor):

        x_tf = X_tensor[:, 0:1]
        t_tf = X_tensor[:, 1:2]
        diff_const = X_tensor[:, 2:3]
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x_tf, t_tf])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x_tf, t_tf])
                X_input = tf.concat([x_tf, t_tf, diff_const], axis=1)  # shape (N, 2)
                u = self.model_cont(X_input)
            u_x = tape1.gradient(u, x_tf)
            u_t = tape1.gradient(u, t_tf)
                
        u_xx = tape2.gradient(u_x, x_tf)
        
        f = u_t - diff_const * u_xx
        return f
    
    def loss_terms(self, X_ic_input, X_bc_input, u_ic, u_bc, X_cp_input):
        
        if rank ==0:
            u_ic_pred = self.model_cont(X_ic_input)
            u_bc_pred = self.model_cont(X_bc_input)

            loss_ic = tf.reduce_mean(tf.square(u_ic - u_ic_pred))
            loss_bc = tf.reduce_mean(tf.square(u_bc - u_bc_pred))
            loss_pde = tf.reduce_mean(tf.square(self.pde_residual(X_cp_input)))
            
            loss = loss_ic + loss_bc + loss_pde
            return loss, loss_ic, loss_bc, loss_pde
        else:
            return None, None, None, None
    
    def train_model(self):
        
        if (rank == 0) and (self.print_loss == 1):
            log_file = open("training_log.txt", "w")
            log_file.write("epoch,loss,loss_ic,loss_bc,loss_pde\n")
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        for epoch in range(self.epoch):
            
            data = pinn_gen_D(self.data_path, self.N_IC, 
                        self.N_BC, self.N_CP, 
                        self.N_diff,
                        self.diffusion_coeff)
            X_ic_input, u_ic = data.IC_data()
            X_bc_input, u_bc = data.BC_data()
            X_cp_input = data.CP_data()

            comm.Barrier()
            if rank == 0:
                print(f"All processes passed the barrier, now tranning. Epoch{epoch}")
                            
                if X_ic_input is None or u_ic is None:
                    raise ValueError("Initial condition data is None")
                elif X_bc_input is None or u_bc is None:
                    raise ValueError("Boundary condition data is None")
                elif X_cp_input is None:
                    raise ValueError("Collocation data is None")
                
                
                with tf.GradientTape() as tape:
                    loss, loss_ic, loss_bc, loss_pde = self.loss_terms(X_ic_input, X_bc_input, u_ic, u_bc, X_cp_input)
                
                grads = tape.gradient(loss, self.model_cont.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model_cont.trainable_variables))
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}: Total Loss = {loss.numpy():.4e} | IC = {loss_ic.numpy():.4e} | BC = {loss_bc.numpy():.4e} | PDE = {loss_pde.numpy():.4e}")
                
                if self.print_loss == 1:
                    log_str = (
                        f"{epoch},{loss.numpy():.6e},{loss_ic.numpy():.6e},"
                        f"{loss_bc.numpy():.6e},{loss_pde.numpy():.6e}\n"
                    )
                    log_file.write(log_str)
                
        if (rank == 0) and (self.print_loss == 1):
            log_file.close()
        
        
    def predict(self, x, t, D):        
        # Ensure x and t are column vectors
        x = np.reshape(x, (-1, 1))
        t = np.reshape(t, (-1, 1))
        
        if np.isscalar(D):
            D = np.full_like(x, D)
        D = np.reshape(D, (-1, 1))
        # Convert to TensorFlow tensors
        x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
        t_tf = tf.convert_to_tensor(t, dtype=tf.float32)
        d_tf = tf.convert_to_tensor(D, dtype=tf.float32)

        # Concatenate inputs
        X_input = tf.concat([x_tf, t_tf, d_tf], axis=1)

        # Make prediction
        u_pred = self.model_cont(X_input)
        return u_pred
    
    def save_model(self, save_path):
        """
        Save the trained model weights to a given path.
        """
        if rank == 0:
            self.model_cont.save_weights(save_path + "/pinn.weights.h5")
            print(f"Model weights saved to {save_path}")

    def load_model(self, load_path):
        self.model_cont = self.build_pinnmodel()
        self.model_cont.load_weights(load_path)
        print(f"Model weights loaded from {load_path}")

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
    
