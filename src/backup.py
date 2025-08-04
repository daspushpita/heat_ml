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