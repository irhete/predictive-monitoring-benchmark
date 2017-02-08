from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np

class IndexBasedExtractor(TransformerMixin):
    
    def __init__(self, cat_cols, num_cols, max_events, fillna=True):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.max_events = max_events
        self.fillna = fillna
        self.columns = None
    
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        # add missing columns if necessary
        if self.columns is None:
            relevant_num_cols = ["%s_%s"%(col, i) for col in self.num_cols for i in range(self.max_events)]
            relevant_cat_col_prefixes = tuple(["%s_%s_"%(col, i) for col in self.cat_cols for i in range(self.max_events)])
            relevant_cols = [col for col in X.columns if col.startswith(relevant_cat_col_prefixes)] + relevant_num_cols
            self.columns = relevant_cols
        else:
            missing_cols = [col for col in self.columns if col not in X.columns]
            for col in missing_cols:
                X[col] = 0
        
        return X[self.columns]