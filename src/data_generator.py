import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input, regularizers
from mpi4py import MPI
    
    
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


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
    
class data_generator_simple_nn():
    """
    Reads 1D PDE solution snapshots from disk and returns X → Y training data,
    optionally skipping time steps and holding out test data.
    """
    def __init__(self, path, holdout, nsteps, dim, csv=1, npz=0, 
                x_res=100, gen=0, skip=0, pos=0, 
                N_diff=1, diffusion_coeff=None,*args, **kwargs):
        
        self.filename = path
        self.dim = dim
        self.csv = csv
        self.npz = npz
        self.nx = x_res
        self.gen = gen
        self.holdout = holdout
        self.skip = skip
        self.nsteps = nsteps
        self.pos = pos
        self.N_diff = len(diffusion_coeff) if diffusion_coeff is not None else N_diff
        self.diffusion_coeff = diffusion_coeff

    def read_1d(self):
        
        if (self.csv):
            files = sorted(
                [f for f in os.listdir(self.filename) if f.startswith('data_') and f.endswith('.csv')],
                key=lambda s: int(s.split('_')[1].split('.')[0])
            )
            x_shape = self.nx
            
            data_0 = pd.read_csv(os.path.join(self.filename, files[0]))
            self.x = data_0["x"].values.astype(np.float32)
            
            nt = len(files)
            u_all = np.zeros((nt, self.nx), dtype=np.float32)
            t_all = np.zeros((nt,), dtype=np.float32)

            for i, file in enumerate(files):
                data = pd.read_csv(os.path.join(self.filename, file))
                u_all[i] = data.filter(like='u_t').values.flatten()
                t_all[i] = data["t"].iloc[0]
                                
        elif (self.npz):
            data = np.load(self.path)
            u_all = data['u'].astype(np.float32)       # (nt, nx)
            t_all = data['time'].astype(np.float32)    # (nt,)
            self.x= data['x'].astype(np.float32)       # (nx,)

            u_all = u_all[::self.skip]
            t_all = t_all[::self.skip]
        
        else:
            raise ValueError ("No datafile format specified!")
        
        nt = u_all.shape[0]
        k  = int(self.nsteps)
        num_samples = nt - k
        
        if self.pos==0:
            # X: (samples, nx, 3) = [u(x,t_n), x, t_n]        
            X_data = np.zeros((num_samples, self.nx, self.dim), dtype=np.float32)
            if self.dim==1:
                X_data[..., 0] = u_all[:num_samples]
            elif self.dim==2:
                X_data[..., 0] = u_all[:num_samples]          # u(x,t_n)
                X_data[..., 1] = self.x[None, :]              # x
            elif self.dim==3:
                X_data[..., 0] = u_all[:num_samples]          # u(x,t_n)
                X_data[..., 1] = self.x[None, :]              # x
                X_data[..., 2] = t_all[:num_samples, None]    # t_n (broadcasted across x)
            else:
                raise ValueError('Dimensions not understood for u(x, t)')
                
            if X_data.shape[-1] != self.dim:
                raise RuntimeError(f"Feature dim mismatch: got {X_data.shape[-1]}, expected {self.dim}")

            # Build multi-step Y
            # Y: (samples, nx, k, 1) = [u(x,t_{n+1}), ..., u(x,t_{n+k})]
            # stack along horizon, transpose to put nx before k, then add channel dim
            Y = np.stack([u_all[i+1:i+1+k] for i in range(num_samples)], axis=0)  # (samples, k, nx)
            Y = np.transpose(Y, (0, 2, 1))[..., None]                              # (samples, nx, k, 1)

        elif self.pos==1:
            # X: (samples, nx, 3) = [u(x,t_n), x, t_n]        
            X_data = np.zeros((num_samples, self.nx, self.dim))
            x_min, x_max = self.x.min(), self.x.max()
            x_norm = (self.x - x_min) / (x_max - x_min + 1e-12)
            
            if self.dim==1:
                d_model = 2
                X_data[..., 0] = u_all[:num_samples]
                pos_enc = Positional_Encoding(x_norm, d_model=d_model).sinosoidal_encoding()
                pe_all = np.broadcast_to(pos_enc, (num_samples, self.nx, d_model))
                X_data = np.concatenate((X_data, pe_all), axis=-1)
                self.dim = 1 + d_model
                
                # Build multi-step Y: (samples, nx, k)
                Y = np.stack([u_all[i+1:i+1+k] for i in range(num_samples)], axis=0)  # (samples, k, nx)
                Y = np.transpose(Y, (0, 2, 1))                                      # (samples, nx, k)
    
            else:
                raise ValueError('Dimensions not understood for u(x, t) with positional encoding')
            
        else:
            raise ValueError('positional encoding flag not understood')
        
        # Split training vs held-out
        if self.holdout > 0:
            if self.holdout >= num_samples:
                raise ValueError(f"holdout ({self.holdout}) must be < samples ({num_samples}).")
            X_train, Y_train = X_data[:-self.holdout], Y[:-self.holdout]
            X_test, Y_test = X_data[-self.holdout:], Y[-self.holdout:]
        else:
            X_train, Y_train = X_data, Y
            X_test, Y_test = None, None 
        
        if self.gen:
            return X_train, Y_train, X_test, Y_test, self.x, t_all
        else:
            return X_train, Y_train, X_test, Y_test, self.x, t_all
        
        
    def read_1d_gen_coeff(self):
        
        training_Ds = self.diffusion_coeff
        training_Ds_norm = (training_Ds - np.min(training_Ds)) / (np.max(training_Ds) - np.min(training_Ds))

        #Xgrid values and time values from one of the diffusion coefficients
        output_dir_0 = os.path.join(self.filename, f"diffusion{str(0.01).replace('.', '.')}/")
        files_0 = sorted(
            [f for f in os.listdir(output_dir_0) if f.startswith('data_') and f.endswith('.csv')],
            key=lambda s: int(s.split('_')[1].split('.')[0])
        )
        data_0 = pd.read_csv(os.path.join(output_dir_0, files_0[0]))
        
        #Get the x grid and positional encodings
        self.x = data_0["x"].values
        x_min, x_max = self.x.min(), self.x.max()
        x_norm = (self.x - x_min) / (x_max - x_min + 1e-12)
        d_model = 2
        pos_enc = Positional_Encoding(x_norm, d_model=d_model).sinosoidal_encoding()
        
        #get the time array
        nt = len(files_0)
        t_all = np.zeros((nt,))
        for i, file in enumerate(files_0):
            data_all = pd.read_csv(os.path.join(output_dir_0, file))
            t_all[i] = data_all["t"].iloc[0]
        

        #################################################################################
        
        X_train_list, Y_train_list = [], []
        X_test_list,  Y_test_list  = [], []
        
        for D, Dn in zip(training_Ds, training_Ds_norm):
            output_dir = os.path.join(self.filename, f"diffusion{str(D).replace('.', '.')}/")
            files = sorted(
                [f for f in os.listdir(output_dir) if f.startswith('data_') and f.endswith('.csv')],
                key=lambda s: int(s.split('_')[1].split('.')[0])
            )
            nt = len(files)
            k  = int(self.nsteps)
            num_samples = nt - k
            
            u_all = np.zeros((nt, self.nx))
            X_data_D = np.zeros((num_samples, self.nx, 1))
            for i, file in enumerate(files):
                data = pd.read_csv(os.path.join(output_dir, file))
                u_all[i] = data.filter(like='u_t').values.flatten()
            
            X_data_D[..., 0] = u_all[:num_samples]
            pe_all = np.broadcast_to(pos_enc[None, ...], (num_samples, self.nx, d_model))
            Dcoeff = np.full((num_samples, self.nx, 1), Dn)
            
            X_data_D = np.concatenate((X_data_D, pe_all, Dcoeff), axis=-1)
            
            # Build multi-step Y: (samples, nx, k)
            Y_data_D = np.stack([u_all[i+1:i+1+k] for i in range(num_samples)], axis=0)  
            Y_data_D = np.transpose(Y_data_D, (0, 2, 1))

            # Split training vs held-out
            if self.holdout > 0:
                if self.holdout >= num_samples:
                    raise ValueError(f"holdout ({self.holdout}) must be < samples ({num_samples}).")
                
                X_train_list.append(X_data_D[:-self.holdout])
                Y_train_list.append(Y_data_D[:-self.holdout])
                X_test_list.append(X_data_D[-self.holdout:])
                Y_test_list.append(Y_data_D[-self.holdout:])
            else:
                X_train_list.append(X_data_D)
                Y_train_list.append(Y_data_D)
            
        # concat across coefficients
        X_train = np.concatenate(X_train_list, axis=0)
        Y_train = np.concatenate(Y_train_list, axis=0)
        
        if self.holdout > 0:
            X_test = np.concatenate(X_test_list, axis=0)
            Y_test = np.concatenate(Y_test_list, axis=0)
        else:
            X_test, Y_test = None, None
        #################################################################################       
        # final feature dim for model
        self.dim = 1 + d_model + 1  # u + PE + ν
        #################################################################################
        if self.gen:
            return X_train, Y_train, X_test, Y_test, self.x, t_all
        else:
            return X_train, Y_train, X_test, Y_test, self.x, t_all

        
        
class pinn_single_D():
        
    def __init__(self, datapath, N_IC, N_BC, N_CP, N_diff,
                 diffusion_coeff):
        
        self.datapath = datapath
        self.N_IC = N_IC
        self.N_BC = N_BC
        self.N_CP = N_CP
        self.N_diff = N_diff
        self.diffusion_coeff = diffusion_coeff
        
    def time_array(self, same_dt_allfiles):

        if (same_dt_allfiles == True):
            data_path = os.path.join(self.datapath, f'diffusion{self.diffusion_coeff:.2f}')
            files = sorted([f for f in os.listdir(data_path) if f.startswith('data_') and f.endswith('.csv')])

            nt = len(files) - 1 #Ignoring the initial condition
            time_array = np.zeros((nt))

            if nt <= 0:
                raise ValueError(f"[Rank {rank}] No usable data files found in {data_path}. Files found: {files}")

            for i, file in enumerate(files[1:]):
                data_all = pd.read_csv(os.path.join(data_path, file))
                time_array[i] = data_all["t"].iloc[0]
            return time_array
        else:
            raise ValueError("data generation not implemented for tranning data with different dt")
        
        
    def IC_data(self):
        data = pd.read_csv(self.datapath+'data_0000.csv')
        x = data['x']
        u_xt = data['u_t0']
        idx = np.random.choice(len(x), size=self.N_IC, replace=False)
        x_ic = x[idx]
        t_ic = np.zeros_like(x_ic)
        u_xt_ic = u_xt[idx]
        x_tf_ic = tf.convert_to_tensor(x_ic, dtype=tf.float32)
        t_tf_ic = tf.convert_to_tensor(t_ic, dtype=tf.float32)
        u_tf_ic = tf.convert_to_tensor(u_xt_ic.to_numpy()[:, None], dtype=tf.float32)
        X_input = tf.concat([x_tf_ic[:, None], t_tf_ic[:, None]], axis=1)
        return X_input, u_tf_ic
        
    def BC_data(self):
        time = self.time_array(same_dt_allfiles=True)
        idt = np.random.choice(len(time), size=self.N_BC, replace=False)
        t_bc = time[idt]

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
            
    def CP_data(self):
        
        time = self.time_array(same_dt_allfiles=True)
        x_cp = np.random.uniform(0.0, 1.0, size=self.N_CP)
        t_cp = np.random.uniform(np.min(time), np.max(time), size=self.N_CP)

        # Convert to tensors
        x_tf_cp = tf.convert_to_tensor(x_cp[:, None], dtype=tf.float32)
        t_tf_cp = tf.convert_to_tensor(t_cp[:, None], dtype=tf.float32)

        X_input = tf.concat([x_tf_cp, t_tf_cp], axis=1)
        return X_input


class pinn_gen_D():
    
    def __init__(self, datapath, N_IC, N_BC, N_CP, N_diff,
                 diffusion_coeff):
        
        self.datapath = datapath
        self.N_IC = N_IC
        self.N_BC = N_BC
        self.N_CP = N_CP
        self.N_diff = N_diff
        self.diffusion_coeff = diffusion_coeff
        
    def time_array(self, same_dt_allfiles):

        if (same_dt_allfiles == True):
            data_path = os.path.join(self.datapath,  f'diffusion0.01')
            files = sorted([f for f in os.listdir(data_path) if f.startswith('data_') and f.endswith('.csv')])

            nt = len(files) - 1 #Ignoring the initial condition
            time_array = np.zeros((nt))

            if nt <= 0:
                raise ValueError(f"[Rank {rank}] No usable data files found in {data_path}. Files found: {files}")

            for i, file in enumerate(files[1:]):
                data_all = pd.read_csv(os.path.join(data_path, file))
                time_array[i] = data_all["t"].iloc[0]
            return time_array
        else:
            raise ValueError("data generation not implemented for tranning data with different dt")
        
    def IC_data(self):
        
        x_list, t_list, d_list, u_list = [], [], [], []
        
        if rank == 0:
            id_D = np.random.choice(self.diffusion_coeff, size=self.N_diff, replace=False)
        else:
            id_D = None
            
        id_D_all = comm.bcast(id_D, root=0)
        split_D = np.array_split(id_D_all, size)
        my_D = split_D[rank]

        if len(my_D) == 0:
            return None, None
        else:
            for D_coeff in my_D:
                data_path = os.path.join(self.datapath,  f'diffusion{D_coeff}')
                data = pd.read_csv(os.path.join(data_path, 'data_0000.csv'))
                
                x = data['x']
                u_xt = data['u_t0']
                idx = np.random.choice(len(x), size=self.N_IC, replace=False)

                x_ic = x[idx].to_numpy()
                u_ic = u_xt[idx].to_numpy()
                t_ic = np.zeros_like(x_ic)
                d_ic = np.full_like(x_ic, D_coeff)

                x_list.append(x_ic[:, None])
                t_list.append(t_ic[:, None])
                d_list.append(d_ic[:, None])
                u_list.append(u_ic[:, None])

            x_tf = tf.convert_to_tensor(np.concatenate(x_list), dtype=tf.float32)
            t_tf = tf.convert_to_tensor(np.concatenate(t_list), dtype=tf.float32)
            diff_tf = tf.convert_to_tensor(np.concatenate(d_list), dtype=tf.float32)
            u_tf = tf.convert_to_tensor(np.concatenate(u_list), dtype=tf.float32)
            X_local = tf.concat([x_tf, t_tf, diff_tf], axis=1)
            u_local = u_tf
            X_all = comm.gather(X_local, root=0)
            u_all = comm.gather(u_local, root=0)
            if rank == 0:
                X_all = tf.concat(X_all, axis=0)
                u_all = tf.concat(u_all, axis=0)
                return X_all, u_all
            else:
                return None, None
            
    def BC_data(self):
        
        x_list, t_list, d_list, u_list = [], [], [], []

        if rank == 0:
            id_D = np.random.choice(self.diffusion_coeff, size=self.N_diff, replace=False)
        else:
            id_D = None
            
        id_D_all = comm.bcast(id_D, root=0)
        D_split = np.array_split(id_D_all, size)
        my_D = D_split[rank]

        if len(my_D) == 0:
            return None, None
        else:
            for D_coeff in my_D:
                time_array = self.time_array(same_dt_allfiles=True)
                n_available = len(time_array)
                if self.N_BC > n_available:
                    raise ValueError(f"Cannot sample {self.N_BC} unique times from only {n_available} available.")
                
                idt = np.random.choice(len(time_array), size=self.N_BC, replace=False)
                t = time_array[idt]

                # Repeat x=0 and x=1 for each sampled time
                x = np.tile([0.0, 1.0], self.N_BC)  # shape (2*N_BC,)
                t = np.repeat(t, 2)             # shape (2*N_BC,)
                u = np.zeros_like(x)            # (e.g., Dirichlet BC: u=0 at both ends)
                diff = np.full_like(x, D_coeff)
                
                x_list.append(x[:, None]) #[:,None] -> makes the shape (2*N_BC,) -> (2*N_BC, 1)
                t_list.append(t[:, None])
                d_list.append(diff[:, None])
                u_list.append(u[:, None])
                
            # Convert to tensors
            x_tf_bc = tf.convert_to_tensor(np.concatenate(x_list), dtype=tf.float32)
            t_tf_bc = tf.convert_to_tensor(np.concatenate(t_list), dtype=tf.float32)
            u_tf_bc = tf.convert_to_tensor(np.concatenate(u_list), dtype=tf.float32)
            diff_tf_bc = tf.convert_to_tensor(np.concatenate(d_list), dtype=tf.float32)

            X_local = tf.concat([x_tf_bc, t_tf_bc, diff_tf_bc], axis=1)
            u_local = u_tf_bc
            X_all = comm.gather(X_local, root=0)
            u_all = comm.gather(u_local, root=0)
            if rank == 0:
                X_all = tf.concat(X_all, axis=0)
                u_all = tf.concat(u_all, axis=0)
                return X_all, u_all
            else:
                return None, None
            
            
    def CP_data(self):

        x_list, t_list, d_list = [], [], []
        if rank == 0:
            id_D = np.random.choice(self.diffusion_coeff, size=self.N_diff, replace=False)
        else:
            id_D = None
            
        id_D_all = comm.bcast(id_D, root=0)
        D_split = np.array_split(id_D_all, size)
        my_D = D_split[rank]

        if len(my_D) == 0:
            return None, None
        else:        
            for D_coeff in my_D:
                time_array = self.time_array(same_dt_allfiles=True)
                n_available = len(time_array)
                if self.N_CP > n_available:
                    raise ValueError(f"Cannot sample {self.N_CP} unique times from only {n_available} available.")
            
                x_cp = np.random.uniform(0.0, 1.0, size=self.N_CP)
                t_cp = np.random.uniform(np.min(time_array), np.max(time_array), size=self.N_CP)
                diff_cp = np.full_like(x_cp, D_coeff)
                
                x_list.append(x_cp[:, None])
                t_list.append(t_cp[:, None])
                d_list.append(diff_cp[:, None])

            # Convert to tensors
            x_tf_cp = tf.convert_to_tensor(np.concatenate(x_list), dtype=tf.float32)
            t_tf_cp = tf.convert_to_tensor(np.concatenate(t_list), dtype=tf.float32)
            diff_tf_cp = tf.convert_to_tensor(np.concatenate(d_list), dtype=tf.float32)
            X_local = tf.concat([x_tf_cp, t_tf_cp, diff_tf_cp], axis=1)
            X_all = comm.gather(X_local, root=0)
            if rank == 0:
                X_all = tf.concat(X_all, axis=0)
                return X_all
            else:
                return None, None