from sklearn.base import TransformerMixin
import pandas as pd

class FrequencyEncoder(TransformerMixin):
    
    def __init__(self, case_id_col, cols_to_encode):
        self.case_id_col = case_id_col
        self.cols_to_encode = cols_to_encode
        self.columns = None
    
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, data, y=None):
        
        # reshape: each activity will be separate column
        data_encoded = pd.get_dummies(data[self.cols_to_encode])
        data_encoded[self.case_id_col] = data[self.case_id_col]
        
        # add missing columns if necessary
        if self.columns is None:
            self.columns = data_encoded.columns
        else:
            missing_cols = [col for col in self.columns if col not in data_encoded.columns]
            for col in missing_cols:
                data_encoded[col] = 0
            data_encoded = data_encoded[self.columns]
        
        # aggregate activities by case
        grouped = data_encoded.groupby(self.case_id_col)
        data = grouped.sum()
        data[self.case_id_col] = data.index
        return data