from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np

class AggregateTransformer(TransformerMixin):
    
    def __init__(self, case_id_col, timestamp_col, cat_cols, num_cols, fillna=True):
        self.case_id_col = case_id_col
        self.timestamp_col = timestamp_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.fillna = fillna
        self.columns = None
    
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        grouped = X.sort_values(by=self.timestamp_col, ascending=True).groupby(self.case_id_col)
        
        # transform numeric cols
        dt_numeric = pd.DataFrame(index=X[self.case_id_col].unique())
        for col in self.num_cols:
            tmp = grouped[col].agg({'%s_mean'%col:np.mean, '%s_max'%col:np.max, '%s_min'%col:np.min, '%s_sum'%col:np.sum, '%s_std'%col:np.std})
            dt_numeric = dt_numeric.merge(tmp, left_index=True, right_index=True)
        
        # transform cat cols
        dt_cat = pd.get_dummies(X[self.cat_cols])
        dt_cat[self.case_id_col] = X[self.case_id_col]
        dt_cat = dt_cat.groupby(self.case_id_col).sum()
        dt_cat[self.case_id_col] = dt_cat.index.get_level_values(self.case_id_col)
        
        # merge
        dt_transformed = dt_cat.merge(dt_numeric, left_index=True, right_index=True)
        dt_transformed[self.case_id_col] = dt_transformed.index.get_level_values(self.case_id_col)
        
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