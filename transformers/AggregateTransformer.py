from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np

class AggregateTransformer(TransformerMixin):
    
    def __init__(self, case_id_col, cat_cols, num_cols, fillna=True):
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.fillna = fillna
        
        self.columns = None
    
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        # transform cat cols
        dt_transformed = pd.get_dummies(X[self.cat_cols])
        dt_transformed[self.case_id_col] = X[self.case_id_col]
        dt_transformed = dt_transformed.groupby(self.case_id_col).sum()
        
        # transform numeric cols
        if len(self.num_cols) > 0:
            dt_numeric = X.groupby(self.case_id_col)[self.num_cols].agg({'mean':np.mean, 'max':np.max, 'min':np.min, 'sum':np.sum, 'std':np.std})
            dt_numeric.columns = ['_'.join(col).strip() for col in dt_numeric.columns.values]
            dt_transformed = pd.concat([dt_transformed, dt_numeric], axis=1)
        
        # fill missing values with 0-s
        if self.fillna:
            dt_transformed.fillna(0, inplace=True)
            
        # add missing columns if necessary
        if self.columns is None:
            self.columns = dt_transformed.columns
        else:
            missing_cols = [col for col in self.columns if col not in dt_transformed.columns]
            for col in missing_cols:
                dt_transformed[col] = 0
            dt_transformed = dt_transformed[self.columns]
            
        return dt_transformed