import pandas as pd
import numpy as np
from time import time
import sys

class StateBasedBucketer(object):
    
    def __init__(self, encoder):
        self.encoder = encoder
        
        self.dt_states = None
        self.n_states = 0
        
    
    def fit(self, X, y=None):
        
        dt_encoded = self.encoder.fit_transform(X)
        
        self.dt_states = dt_encoded.drop_duplicates()
        self.dt_states = self.dt_states.assign(state = range(len(self.dt_states)))
        
        self.n_states = len(self.dt_states)
        
        return self
    
    
    def predict(self, X, y=None):
        
        dt_encoded = self.encoder.transform(X)
        
        dt_transformed = pd.merge(dt_encoded, self.dt_states, how='left')
        dt_transformed.fillna(-1, inplace=True)
        
        return dt_transformed["state"].astype(int).as_matrix()
    
    
    def fit_predict(self, X, y=None):
        
        self.fit(X)
        return self.predict(X)