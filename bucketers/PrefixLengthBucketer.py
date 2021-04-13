import pandas as pd
import numpy as np
from time import time
import sys

class PrefixLengthBucketer(object):
    
    def __init__(self, case_id_col):
        self.n_states = 0
        self.case_id_col = case_id_col
        
    
    def fit(self, X, y=None):
        
        sizes = X.groupby(self.case_id_col).size()
        self.n_states = sizes.unique()
        
        return self
    
    
    def predict(self, X, y=None):
        
        bucket_assignments = X.groupby(self.case_id_col).size()
        while sum(~bucket_assignments.isin(self.n_states)) > 0:
            bucket_assignments[~bucket_assignments.isin(self.n_states)] -= 1
        return bucket_assignments.to_numpy()
    
    
    def fit_predict(self, X, y=None):
        
        self.fit(X)
        return self.predict(X)