import numpy as np
import time

class ClassifierWrapper(object):
    
    def __init__(self, cls, min_cases_for_training = 30):
        self.cls = cls
        
        self.min_cases_for_training = min_cases_for_training
        self.hardcoded_prediction = None
        
        self.fit_time = None
        self.predict_time = None
        
        
    def fit(self, X, y):
        start = time.time()
        # if there are too few training instances, use the mean
        if X.shape[0] < self.min_cases_for_training:
            self.hardcoded_prediction = np.mean(y)

        # if all the training instances are of the same class, use this class as prediction
        elif len(set(y)) < 2:
            self.hardcoded_prediction = int(y[0])

        else:
            self.cls.fit(X, y)
        self.fit_time = time.time() - start
        return self
    
    
    def predict_proba(self, X, y=None):
        start = time.time()
        if self.hardcoded_prediction is not None:
            self.predict_time = time.time() - start
            return [self.hardcoded_prediction] * X.shape[0]
                        
        else:
            preds_pos_label_idx = np.where(self.cls.classes_ == 1)[0][0] 
            preds = self.cls.predict_proba(X)[:,preds_pos_label_idx]
            self.predict_time = time.time() - start
            return preds
        
    
    def fit_predict(self, X, y):
        
        self.fit(X, y)
        return self.predict_proba(X)