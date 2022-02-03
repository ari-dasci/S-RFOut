from multipledispatch import dispatch
from multipledispatch.variadic import Variadic

from numpy import linalg as LA
import numpy as np

import shfl

class RFOut1d(shfl.federated_aggregator.FedAvgAggregator):
    
    def __init__(self, clip=0, noise_mult=0):
        self._noise = noise_mult
        self._clip = clip
        super().__init__()
    
    @dispatch(Variadic[np.ndarray, np.ScalarType])
    def _aggregate(self, *params):
        clients_params = np.array(params).T
        
        for i, v in enumerate(clients_params):
            mu = np.mean(v)
            std = np.std(v)
            
            for param_i, param_v in enumerate(clients_params[i]):
                if abs(param_v - mu) > 3*std:
                    v[param_i] = mu
        
        clients_params = clients_params.T
        
        for i, v in enumerate(clients_params):
            norm = LA.norm(v)
            clients_params[i] = np.multiply(v, min(1, self._clip/norm))
        
            mean = np.mean(clients_params, axis=0)
        else:
            mean = np.mean(clients_params, axis = 0)
            
        noise = np.random.normal(loc=0.0, scale=self._noise*self._clip/len(clients_params), size=mean.shape) 
        return mean + noise
    
    @dispatch(Variadic[list])
    def _aggregate(self, *params):
        return super()._aggregate(*params)
