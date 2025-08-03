import numpy as np
import os
import pandas as pd


class heat_solver_1d:
    """Finite Difference solver for the heat Equation.
    """
    def __init__(self, x1min, x1max, nx, tmax, cfl, diff_cons, output_dir, npz_file, t_save_int, *args, **kwargs):
        
        self.x1min = x1min
        self.x1max = x1max
        self.nx = nx
        self.cfl = cfl
        self.tmax = tmax
        self.x = np.linspace(self.x1min, self.x1max, self.nx)
        self.dx = (self.x1max - self.x1min) / (self.nx - 1)
        self.diff_cons = diff_cons
        self.output_dir = output_dir
        self.npz_file = npz_file
        self.t_save_int = t_save_int
    
        if not self.npz_file:
            if output_dir:
                self.output_dir = os.path.join(output_dir, f"diffusion{self.diff_cons}")
            else:
                self.output_dir = f"../data/diffusion{self.diff_cons}"
                
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            self.output_dir = None
        
    def initial_condition(self):
        u0 = np.sin(np.pi * self.x)
        return u0
    
    def evolution_euler(self):
        
        t = 0.0
        dt = self.cfl * self.dx**2 / self.diff_cons
        n_steps = int(self.tmax / dt)
        
        u0 = self.initial_condition()
        u_sol = np.zeros((self.x.shape))
        u_sol[:] = u0[:]
        u_sol_new = np.zeros((self.x.shape))
        u_store = np.zeros((n_steps+1, self.nx))
        u_store[0,:] = u_sol[:]
        time_array = [0]
        step = 0
        next_save_time = 0.0
        
        if not self.npz_file:
            # Save initial condition at t = 0
            snapshot_filename = os.path.join(self.output_dir, f"data_{step:04d}.csv")
            df_snapshot = pd.DataFrame({'x': self.x, f'u_t{0}': u_sol,'t': t})
            df_snapshot.to_csv(snapshot_filename, index=False)

        for it in range (0, n_steps, 1):
            #Applying the Boundary conditions
            u_sol_new[0] = 0.0
            u_sol_new[-1] = 0.0
            
            u_sol_new[1:-1] = u_sol[1:-1] + (self.diff_cons * dt / self.dx**2) * (u_sol[0:-2] - \
                                            2.0 * u_sol[1:-1] + u_sol[2:])
        
            #Update the u_sol with the new t value
            t += dt
            u_sol = u_sol_new.copy()

            if t >= next_save_time - 1e-10:
                step += 1
                time_array.append(t)
                u_store[step,:] = u_sol
                
                if not self.npz_file:
                    # Save current snapshot as CSV
                    snapshot_filename = os.path.join(self.output_dir, f"data_{step:04d}.csv")
                    df_snapshot = pd.DataFrame({'x': self.x, f'u_t{step}': u_sol, 't': t})
                    df_snapshot.to_csv(snapshot_filename, index=False)
                
                next_save_time += self.t_save_int

        saved_steps = len(time_array)
        u_store = u_store[:saved_steps, :]
        time_array = np.array(time_array, dtype=float)
        time_column = time_array[:, None]
        u_with_time = np.hstack((time_column, u_store))
        
        if self.npz_file:
            np.savez("../data/heat_eq_euler.npz", u=u_with_time, x=self.x, time=time_array, 
                            nx=self.nx, cfl=self.cfl, diff_cons=self.diff_cons)
        
        return self.x, u_with_time, time_array