import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.base import TransformerMixin

class IndexBasedTransformer(TransformerMixin):
    
    def __init__(self, nr_events, static_cols, dynamic_cols, cat_cols, case_id_col, timestamp_col, label_col, activity_col, fillna=True):
        self.nr_events = nr_events
        self.static_cols = static_cols
        self.dynamic_cols = dynamic_cols + [activity_col]
        self.cat_cols = cat_cols
        self.case_id_col = case_id_col
        self.label_col = label_col
        self.timestamp_col = timestamp_col
        
        self.fillna = fillna
        self.columns = None
        
    
    
    def fit(self, X, y=None):
        return self
    
    
    def transform(self, X):
        
        grouped = X.sort_values([self.timestamp_col],ascending=True).groupby(self.case_id_col, as_index=False)
        # encode static cols
        data_final = grouped.first()[[self.case_id_col, self.label_col] + self.static_cols]
        
        # encode dynamic cols
        for i in range(self.nr_events):
            data_selected = grouped.nth(i)[[self.case_id_col] + self.dynamic_cols]
            data_selected.columns = [self.case_id_col] + ["%s_%s"%(col, i) for col in self.dynamic_cols]
            data_final = pd.merge(data_final, data_selected, on=self.case_id_col, how="left")
        
        # encode cat cols
        all_cat_cols = ["%s_%s"%(col, i) for col in set(self.dynamic_cols).intersection(self.cat_cols) for i in range(self.nr_events)] + [col for col in self.static_cols if col in self.cat_cols]
        
        cat_data = pd.get_dummies(data_final[all_cat_cols])
        
        data_final = data_final[[col for col in data_final.columns if col not in all_cat_cols]]
        data_final = pd.concat([data_final, cat_data], axis=1)
        
        # fill missing values with 0-s
        if self.fillna:
            data_final.fillna(0, inplace=True)
            
        # add missing columns if necessary
        if self.columns is None:
            self.columns = data_final.columns
        else:
            missing_cols = [col for col in self.columns if col not in data_final.columns]
            for col in missing_cols:
                data_final[col] = 0
            data_final = data_final[self.columns]

        return(data_final)
        