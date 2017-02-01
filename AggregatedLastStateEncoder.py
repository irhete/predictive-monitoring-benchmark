from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np

class AggregatedLastStateEncoder(TransformerMixin):
    
    def __init__(self, case_id_col, timestamp_col, numeric_cols, cat_cols, last_state_cols):
        self.case_id_col = case_id_col
        self.numeric_cols = numeric_cols
        self.cat_cols = cat_cols
        self.last_state_cols = last_state_cols
        self.timestamp_col = timestamp_col
        self.columns = None
    
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        grouped = X.sort_values(by=self.timestamp_col, ascending=True).groupby(self.case_id_col)
        
        # encode numeric
        dt_numeric = pd.DataFrame(index=X[self.case_id_col].unique())
        for col in self.numeric_cols:
            if grouped.size().max() > 1:
                tmp = grouped[col].agg({'%s_mean'%col:np.mean, '%s_max'%col:np.max, '%s_min'%col:np.min, '%s_sum'%col:np.sum, '%s_std'%col:np.std, '%s_last'%col:"last", '%s_prev'%col:lambda x: x[-2]})
            else:
                tmp = grouped[col].agg({'%s_mean'%col:np.mean, '%s_max'%col:np.max, '%s_min'%col:np.min, '%s_sum'%col:np.sum, '%s_std'%col:np.std, '%s_last'%col:"last", '%s_prev'%col:0})
            dt_numeric = dt_numeric.merge(tmp, left_index=True, right_index=True)
            
        # encode categorical
        dt_cat = pd.get_dummies(X[self.cat_cols])
        dt_cat[self.case_id_col] = X[self.case_id_col]
        dt_cat = dt_cat.groupby(self.case_id_col).sum()

        dt_cat[self.case_id_col] = dt_cat.index.get_level_values(self.case_id_col)
        for col in self.cat_cols:
            if grouped.size().max() > 1:
                tmp_lasts = grouped[col].agg({'%s_last'%col:"last", '%s_prev'%col:lambda x: x[-2]})
            else:
                tmp_lasts = grouped[col].agg({'%s_last'%col:"last", '%s_prev'%col:"start"})
            dt_cat = dt_cat.merge(pd.get_dummies(tmp_lasts), left_index=True, right_index=True)
            
        # encode last state cols
        dt_last_state = grouped.apply(lambda x: x.tail(1)[self.last_state_cols])
        
        # merge
        data_encoded = dt_cat.merge(dt_numeric, left_index=True, right_index=True)
        data_encoded = data_encoded.merge(dt_last_state, left_index=True, right_index=True)
        data_encoded[self.case_id_col] = data_encoded.index.get_level_values(self.case_id_col)
        
        # add missing columns if necessary
        if self.columns is None:
            self.columns = data_encoded.columns
        else:
            missing_cols = [col for col in self.columns if col not in data_encoded.columns]
            for col in missing_cols:
                data_encoded[col] = 0
            data_encoded = data_encoded[self.columns]
        
        data_encoded.fillna(0, inplace=True)
        
        return data_encoded