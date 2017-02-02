from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np

class IndexBasedTransformer(TransformerMixin):
    
    def __init__(self, case_id_col, timestamp_col, cat_cols, num_cols, nr_events, fillna=True):
        self.case_id_col = case_id_col
        self.timestamp_col = timestamp_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.nr_events = nr_events
        self.fillna = fillna
        self.columns = None
    
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        grouped = X.sort_values(by=self.timestamp_col, ascending=True).groupby(self.case_id_col, as_index=False)
        
        dt_transformed = pd.DataFrame({self.case_id_col:X[self.case_id_col].unique()})
        
        for i in range(self.nr_events):
            dt_index = grouped.nth(i)[[self.case_id_col] + self.cat_cols + self.num_cols]
            dt_index.columns = [self.case_id_col] + ["%s_%s"%(col, i) for col in self.cat_cols] + ["%s_%s"%(col, i) for col in self.num_cols]
            dt_transformed = pd.merge(dt_transformed, dt_index, on=self.case_id_col, how="left")
        
        # encode cat cols
        all_cat_cols = ["%s_%s"%(col, i) for col in self.cat_cols for i in range(self.nr_events)]
        
        dt_cat = pd.get_dummies(dt_transformed[all_cat_cols])
        
        dt_transformed = dt_transformed[[col for col in dt_transformed.columns if col not in all_cat_cols]]
        dt_transformed = pd.concat([dt_transformed, dt_cat], axis=1)
       
        
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