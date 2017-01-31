from sklearn.base import TransformerMixin
import pandas as pd

class LastStateEncoder(TransformerMixin):
    
    def __init__(self, case_id_col, timestamp_col, cat_cols, numeric_cols, fillna=True):
        self.case_id_col = case_id_col
        self.timestamp_col = timestamp_col
        self.cat_cols = cat_cols
        self.numeric_cols = numeric_cols
        self.fillna = fillna
        self.columns = None
    
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, data, y=None):
        
        # reshape: each activity will be separate column
        if len(self.cat_cols) > 0:
            data_cat = pd.get_dummies(data[self.cat_cols])
            data = pd.concat([data[[self.case_id_col,self.timestamp_col]+self.numeric_cols], data_cat], axis=1)
        else:
            data = data[[self.case_id_col,self.timestamp_col]+self.numeric_cols]
        
        # aggregate activities by case
        grouped = data.groupby(self.case_id_col)
        
        # extract values from last event
        data = grouped.apply(lambda x: x.sort_values(by=self.timestamp_col, ascending=True).tail(1)).drop(self.timestamp_col, axis=1)
        
        # fill missing values with 0-s
        if self.fillna:
            data.fillna(0, inplace=True)
            
        # add missing columns if necessary
        if self.columns is None:
            self.columns = data.columns
        else:
            missing_cols = [col for col in self.columns if col not in data.columns]
            for col in missing_cols:
                data[col] = 0
            data = data[self.columns]
            
        return data