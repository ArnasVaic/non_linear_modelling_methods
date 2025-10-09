from dataclasses import dataclass

import numpy as np

@dataclass
class SolverConfig:
    N: int # number of points -1
    T: float # total sim time
    tau: float # time step
    
    @property
    def total_points(self):
        return self.N + 1
    
    @property
    def total_time_steps(self):
        return int(self.T / self.tau)
    

def load_solution(params: SolverConfig, path):

    # Read binary file as complex128
    # (same layout as std::complex<double>)
    u = np.fromfile(path, dtype=np.complex128)

    if u.size % params.total_points == params.total_time_steps:
        new_size = (params.total_time_steps, params.total_points)
    else:
        new_size = (u.size // params.total_points, params.total_points)
    
    u =  np.reshape(u, new_size)

    print(u.shape)
    return u
