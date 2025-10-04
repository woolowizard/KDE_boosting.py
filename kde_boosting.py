import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class KDE_boosting:
    
    def __init__(self):
        pass
    
    def compute_h(self) -> float:
        ''' TBD '''
        pass
        
    def compute_f(self, h: float, x: float, weights: np.array, index: int, X: np.array) -> float:
                    
        ''' function that return density estimation for a generic point x 
            h: bandwith
            x: generic point 
            weights: weights of boosting model 
            X: feature matrix '''          
              
        return (weights[index]/h) * self.gaussian_kernel((x-X[index])/h)
    
        
    def predict(self, x: float, log: bool) -> int:
        ''' TBD '''
        pass
        
    def gaussian_kernel(self, u: float) -> float:
        
        ''' function that return kernel estimation (gaussian) of data point x
            u: transformation [(xi - X)/h] '''
        
        return (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * u**2)  

    def predict(self, x, log=False) -> float:
        
        ''' predict method take in input a data point and return the classification predicted label'''
        
        delta_m = 10 ** (-2)
        n=len(X)
        weights = np.full(n, 1/n)
        M = 2 # number of boost
        
        for m in range(1, M+1):
            f2x_est, f1x_est = 10 ** (-2), 10 ** (-2)
            f1_x_i, f2_x_i = 10 ** (-2), 10 ** (-2)
            
            ''' (1) Obtain a weighted Kernel estimate f_hat_j_m '''
            for i in range(n):
                if y[i] == 1:
                    f1x_est += self.compute_f(h, x, weights, index=i)
                elif y[i] == -1:
                    f2x_est += self.compute_f(h, x, weights, index=i)
            
            ''' (2) Calculate delta_m quantity '''
            pmx = f1x_est / (f1x_est + f2x_est)
            pmx = np.clip(pmx, 1e-10, 1-1e-10)  # evita 0 o 1
            delta_m += 0.5 * np.log(pmx/(1-pmx))

            ''' (3) Update weights '''
            for i in range(n):
                
                if y[i] == 1:
                    f1_x_i += self.compute_f(h, X[i], weights, index=i)
                elif y[i] == -1:
                    f2_x_i += self.compute_f(h, X[i], weights, index=i)
                
                pmx_i = f1_x_i / (f1_x_i + f2_x_i)
                pmx_i = np.clip(pmx_i, 1e-10, 1-1e-10)
                delta_m_x_i = 0.5 * np.log(pmx_i/(1-pmx_i))
                weights[i] *= np.exp(delta_m_x_i * y[i])

            if log:
                print(f"=== iteration {m} weights ===")
                print(weights, '\n') 

        return np.sign(delta_m)