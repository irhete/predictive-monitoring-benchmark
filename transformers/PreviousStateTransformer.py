from sklearn.base import TransformerMixin
import pandas as pd
from time import time

class PreviousStateTransformer(TransformerMixin):
    
    def __init__(self, case_id_col, cat_cols, num_cols, fillna=True):
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.fillna = fillna
        
        self.columns = None
        
        self.fit_time = 0
        self.transform_time = 0
        
    
    def fit(self, X, y=None):
        return self
    
    
    def transform(self, X, y=None):
        start = time()
        
        dt_last = X.groupby(self.case_id_col).nth(-2)
        
        # transform numeric cols
        dt_transformed = dt_last[self.num_cols]
        
        # transform cat cols
        if len(self.cat_cols) > 0:
            dt_cat = pd.get_dummies(dt_last[self.cat_cols])
            dt_transformed = pd.concat([dt_transformed, dt_cat], axis=1)

        # add 0 rows where previous value did not exist
        dt_transformed = dt_transformed.reindex(X.groupby(self.case_id_col).first().index, fill_value=0)
            
        # fill NA with 0 if requested
        if self.fillna:
            dt_transformed.fillna(0, inplace=True)
            
        # add missing columns if necessary
        if self.columns is not None:
            missing_cols = [col for col in self.columns if col not in dt_transformed.columns]
            for col in missing_cols:
                dt_transformed[col] = 0
            dt_transformed = dt_transformed[self.columns]
        else:
            self.columns = dt_transformed.columns
        
        self.transform_time = time() - start
        return dt_transformed
    
    
