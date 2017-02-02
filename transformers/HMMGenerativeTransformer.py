import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from hmmlearn import hmm

class HMMGenerativeTransformer(TransformerMixin):
    
    def __init__(self, n_states, num_cols, cat_cols, case_id_col, timestamp_col, min_seq_length=2, max_seq_length=None, random_state=None, n_iter=10, fillna=True):
        
        self.hmms = None
        self.encoders = None
        
        self.case_id_col = case_id_col
        self.timestamp_col = timestamp_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        
        self.n_states = n_states
        self.n_iter = n_iter
        self.min_seq_length = min_seq_length
        self.random_state = random_state
        
        if max_seq_length is None:
            self.max_seq_length = 10000000
        else:
            self.max_seq_length = max_seq_length
            
        self.fillna = fillna
        self.columns = None
    
    
    def fit(self, X, y=None):
        
        self.hmms, self.encoders = self._train_hmms(X)
        
        return self
    
    
    def transform(self, X, y=None):
        grouped = X.sort_values(by=self.timestamp_col, ascending=True).groupby(self.case_id_col)
        scores = grouped.apply(self._calculate_scores)
        dt_scores = pd.DataFrame.from_records(list(scores.values), columns=["hmm_%s_state%s"%(col, state) for col in self.cat_cols + self.num_cols for state in range(self.n_states)])
        dt_scores[self.case_id_col] = scores.index
        
        # fill missing values with 0-s
        if self.fillna:
            dt_scores.fillna(0, inplace=True)
            
        # add missing columns if necessary
        if self.columns is None:
            self.columns = dt_scores.columns
        else:
            missing_cols = [col for col in self.columns if col not in dt_scores.columns]
            for col in missing_cols:
                dt_scores[col] = 0
            dt_scores = dt_scores[self.columns]
            
        return dt_scores
        
        
    def _train_hmms(self, X):

        grouped = X.groupby(self.case_id_col)
        hmms = {}
        encoders = {}

        for col in self.cat_cols + self.num_cols:
            tmp_dt_hmm = []
            for name, group in grouped:
                if len(group) >= self.min_seq_length:
                    seq = [val for val in group.sort_values(self.timestamp_col, ascending=1)[col]]
                    if self.max_seq_length is not None:
                        seq = seq[:self.max_seq_length]
                    tmp_dt_hmm.extend(seq)
            if col in self.cat_cols:
                hmms[col] = hmm.MultinomialHMM(n_components=self.n_states, random_state=self.random_state, n_iter=self.n_iter)
                encoders[col] = {label:idx for idx, label in enumerate(set(tmp_dt_hmm))}
                tmp_dt_hmm = [encoders[col][label] for label in tmp_dt_hmm]
                
            else:
                hmms[col] = hmm.GaussianHMM(n_components=self.n_states, random_state=self.random_state, n_iter=self.n_iter)
                
            hmms[col] = hmms[col].fit(np.atleast_2d(tmp_dt_hmm).T, [min(val, self.max_seq_length) for val in grouped.size() if val >= self.min_seq_length])
            
            # check for buggy transition matrix
            rowsums = hmms[col].transmat_.sum(axis=1)
            if rowsums.sum() < len(rowsums):
                for problem_row_idx in np.where(rowsums < 1)[0]:
                    hmms[col].transmat_[problem_row_idx] = 1.0 / hmms[col].transmat_.shape[1]
                

        return (hmms, encoders)
    
    
    def _calculate_scores(self, group):

        if len(group) < self.min_seq_length:
            return tuple([0] * (len(self.cat_cols) + len(self.num_cols)))

        scores = []

        for col in self.cat_cols + self.num_cols:
            tmp_dt_hmm = [val for val in group.sort_values(self.timestamp_col, ascending=1)[col]]
            if self.max_seq_length is not None:
                tmp_dt_hmm = tmp_dt_hmm[:self.max_seq_length]

            if col in self.cat_cols:

                tmp_dt_hmm = [self.encoders[col][label] for label in tmp_dt_hmm if label in self.encoders[col]]

                score = [0] * self.n_states
                if len(tmp_dt_hmm) >= self.min_seq_length:
                    score = self.hmms[col].predict_proba(np.atleast_2d(tmp_dt_hmm).T)

            else:
                score = self.hmms[col].predict_proba(np.atleast_2d(tmp_dt_hmm).T)
            
            scores.extend(score[-1])

        return tuple(scores)
           