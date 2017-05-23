import pandas as pd
import numpy as np
from time import time
import sys

class NoBucketer(object):
    
    def __init__(self, case_id_col):
        self.n_states = 1
        self.case_id_col = case_id_col
        
    
    def fit(self, X, y=None):
        
        return self
    
    
    def predict(self, X, y=None):
        
        return np.ones(len(X[self.case_id_col].unique()), dtype=np.int)
    
    
    def fit_predict(self, X, y=None):
        
        self.fit(X)
        return self.predict(X)