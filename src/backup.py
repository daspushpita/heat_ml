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
        
        
    def pinn_data_generator_gen(self, IC, BC, CP):
        
        if IC==True:
            x_list_ic = []
            t_list_ic = []
            d_list_ic = []
            u_list_ic = []
            
            id_D = np.random.choice(self.diffusion_coeff, size=self.N_diff, replace=False)
            
            for D_coeff in id_D:
                datapath = self.data_path+f'/diffusion{D_coeff}/'%D_coeff
                data = pd.read_csv(datapath+'/data_0000.csv')
                
                x = data['x']
                u_xt = data['u_t0']
                idx = np.random.choice(len(x), size=self.N_IC, replace=False)
                
                x_ic = x[idx]
                t_ic = np.zeros_like(x_ic)
                u_xt_ic = u_xt[idx]
                diff_ic = np.full_like(x_ic, D_coeff)

                x_list_ic.append(x_ic.to_numpy()[:, None])
                t_list_ic.append(t_ic[:, None])
                d_list_ic.append(diff_ic[:, None])
                u_list_ic.append(u_xt_ic.to_numpy()[:, None])

            x_tf_ic = tf.convert_to_tensor(np.vstack(x_list_ic), dtype=tf.float32)
            t_tf_ic = tf.convert_to_tensor(np.vstack(t_list_ic), dtype=tf.float32)
            diff_tf_ic = tf.convert_to_tensor(np.vstack(d_list_ic), dtype=tf.float32)
            u_tf_ic = tf.convert_to_tensor(np.vstack(u_list_ic), dtype=tf.float32)
            X_input = tf.concat([x_tf_ic, t_tf_ic, diff_tf_ic], axis=1)
            return X_input, u_tf_ic
        
        else:
            if BC == True:
                x_list_bc = []
                t_list_bc = []
                d_list_bc = []
                u_list_bc = []
                
                id_D = np.random.choice(self.diffusion_coeff, size=self.N_diff, replace=False)
                
                for D_coeff in id_D:
                    datapath = self.data_path+f'/diffusion{D_coeff}/'%D_coeff
                    files = sorted([f for f in os.listdir(datapath) if f.startswith('data_') and f.endswith('.csv')])
                    nt = len(files) - 1 #Ignoring the initial condition
                    time_array = np.zeros((nt))
                    for i, file in enumerate(files[1:]):
                        data_all = pd.read_csv(os.path.join(datapath, file))
                        var_t = data_all["t"].iloc[0]
                        time_array[i] = var_t

                    idt = np.random.choice(len(time_array), size=self.N_BC, replace=False)
                    t_bc = time_array[idt]

                    # Repeat x=0 and x=1 for each sampled time
                    x_bc = np.tile([0.0, 1.0], self.N_BC)  # shape (2*N_BC,)
                    t_bc = np.repeat(t_bc, 2)             # shape (2*N_BC,)
                    u_bc = np.zeros_like(x_bc)            # (e.g., Dirichlet BC: u=0 at both ends)
                    diff_bc = np.full_like(x_bc, D_coeff)
                    
                    x_list_bc.append(x_bc[:, None])
                    t_list_bc.append(t_bc[:, None])
                    d_list_bc.append(diff_bc[:, None])
                    u_list_bc.append(u_bc[:, None])
                    
                # Convert to tensors
                x_tf_bc = tf.convert_to_tensor(np.vstack(x_list_bc), dtype=tf.float32)
                t_tf_bc = tf.convert_to_tensor(np.vstack(t_list_bc), dtype=tf.float32)
                u_tf_bc = tf.convert_to_tensor(np.vstack(u_list_bc), dtype=tf.float32)
                diff_tf_bc = tf.convert_to_tensor(np.vstack(d_list_bc), dtype=tf.float32)

                X_input = tf.concat([x_tf_bc, t_tf_bc, diff_tf_bc], axis=1)
                return X_input, u_tf_bc
            
            elif CP == True:
                
                x_list_cp = []
                t_list_cp = []
                d_list_cp = []
                
                id_D = np.random.choice(self.diffusion_coeff, size=self.N_diff, replace=False)
                
                for D_coeff in id_D:
                    datapath = self.data_path+f'/diffusion{D_coeff}/'%D_coeff
                    files = sorted([f for f in os.listdir(datapath) if f.startswith('data_') and f.endswith('.csv')])
                    nt = len(files) - 1 #Ignoring the initial condition
                    time_array = np.zeros((nt))
                    for i, file in enumerate(files[1:]):
                        data_all = pd.read_csv(os.path.join(datapath, file))
                        var_t = data_all["t"].iloc[0]
                        time_array[i] = var_t             
                
                    x_cp = np.random.uniform(0.0, 1.0, size=self.N_CP)
                    t_cp = np.random.uniform(np.min(time_array), np.max(time_array), size=self.N_CP)
                    diff_cp = np.full_like(x_cp, D_coeff)
                    
                    x_list_cp.append(x_cp[:, None])
                    t_list_cp.append(t_cp[:, None])
                    d_list_cp.append(diff_cp[:, None])

                # Convert to tensors
                x_tf_cp = tf.convert_to_tensor(np.vstack(x_list_cp), dtype=tf.float32)
                t_tf_cp = tf.convert_to_tensor(np.vstack(t_list_cp), dtype=tf.float32)
                diff_tf_cp = tf.convert_to_tensor(np.vstack(d_list_cp), dtype=tf.float32)

                X_input = tf.concat([x_tf_cp, t_tf_cp, diff_tf_cp], axis=1)
                return X_input
            
            else:
                raise ValueError("Train Data type not specified")