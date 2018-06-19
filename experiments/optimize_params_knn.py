import EncoderFactory
from DatasetManager import DatasetManager
import BucketFactory

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


import time
import os
import sys
from sys import argv
import pickle
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample


def create_and_evaluate_model(args):
    global trial_nr
    trial_nr += 1
    
    start = time.time()
    score = 0
    for cv_iter in range(n_splits):
        
        dt_test_prefixes = dt_prefixes[cv_iter]
        dt_train_prefixes = pd.DataFrame()
        for cv_train_iter in range(n_splits): 
            if cv_train_iter != cv_iter:
                dt_train_prefixes = pd.concat([dt_train_prefixes, dt_prefixes[cv_train_iter]], axis=0)
        
        # Bucketing prefixes based on control flow
        knn_encoder_args = {'case_id_col':dataset_manager.case_id_col, 
                            'static_cat_cols':[],
                            'static_num_cols':[], 
                            'dynamic_cat_cols':[dataset_manager.activity_col],
                            'dynamic_num_cols':[], 
                            'fillna':True}
        # initiate the KNN model
        bucket_encoder = EncoderFactory.get_encoder(bucket_encoding, **knn_encoder_args)
        encoded_train = bucket_encoder.fit_transform(dt_train_prefixes)
        bucketer = NearestNeighbors(n_neighbors=args["n_neighbors"], algorithm='auto').fit(encoded_train)

        # get nearest neighbors for each test case
        encoded_test = bucket_encoder.fit_transform(dt_test_prefixes)
        _, indices = bucketer.kneighbors(encoded_test)
        
        preds_all = []
        test_y_all = []
        for i in range(len(encoded_test)):

            # retrieve nearest neighbors from training set
            knn_idxs = indices[i]
            relevant_cases_bucket = encoded_train.iloc[knn_idxs].index
            dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes, relevant_cases_bucket) # one row per event
            train_y = dataset_manager.get_label_numeric(dt_train_bucket)
            
            # select current test case
            relevant_test_case = [encoded_test.index[i]]
            dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_test_case)
            test_y_all.extend(dataset_manager.get_label_numeric(dt_test_bucket))
            
            if len(set(train_y)) < 2:
                preds_all.append(train_y[0])
            else:
                feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])

                if cls_method == "rf":
                    cls = RandomForestClassifier(n_estimators=500,
                                                 max_features=args['max_features'],
                                                 random_state=random_state)

                elif cls_method == "xgboost":
                    cls = xgb.XGBClassifier(objective='binary:logistic',
                                            n_estimators=500,
                                            learning_rate= args['learning_rate'],
                                            subsample=args['subsample'],
                                            max_depth=int(args['max_depth']),
                                            colsample_bytree=args['colsample_bytree'],
                                            min_child_weight=int(args['min_child_weight']),
                                            seed=random_state)

                elif cls_method == "logit":
                        cls = LogisticRegression(C=2**args['C'],
                                                 random_state=random_state)

                elif cls_method == "svm":
                    cls = SVC(C=2**args['C'],
                              gamma=2**args['gamma'],
                              random_state=random_state)

                if cls_method == "svm" or cls_method == "logit":
                    pipeline = Pipeline([('encoder', feature_combiner), ('scaler', StandardScaler()), ('cls', cls)])
                else:
                    pipeline = Pipeline([('encoder', feature_combiner), ('cls', cls)])
                pipeline.fit(dt_train_bucket, train_y)

                if cls_method == "svm":
                    preds = pipeline.decision_function(dt_test_bucket)
                else:
                    preds_pos_label_idx = np.where(cls.classes_ == 1)[0][0]
                    preds = pipeline.predict_proba(dt_test_bucket)[:,preds_pos_label_idx]
                    
                preds_all.extend(preds)

        score += roc_auc_score(test_y_all, preds_all)
    
    for k, v in args.items():
        fout_all.write("%s;%s;%s;%s;%s;%s;%s\n" % (trial_nr, dataset_name, cls_method, method_name, k, v, score / n_splits))   
    fout_all.write("%s;%s;%s;%s;%s;%s;%s\n" % (trial_nr, dataset_name, cls_method, method_name, "processing_time", time.time() - start, 0))   
    fout_all.flush()
    return {'loss': -score / n_splits, 'status': STATUS_OK, 'model': cls}


dataset_ref = argv[1]
params_dir = argv[2]
n_iter = int(argv[3])
bucket_method = argv[4]
cls_encoding = argv[5]
cls_method = argv[6]

if bucket_method == "state":
    bucket_encoding = "last"
else:
    bucket_encoding = "agg"

method_name = "%s_%s"%(bucket_method, cls_encoding)

dataset_ref_to_datasets = {
    "bpic2011": ["bpic2011_f%s"%formula for formula in range(1,5)],
    "bpic2015": ["bpic2015_%s_f2"%(municipality) for municipality in range(1,6)],
    "insurance": ["insurance_activity", "insurance_followup"],
    "sepsis_cases": ["sepsis_cases_1", "sepsis_cases_2", "sepsis_cases_4"]
}

encoding_dict = {
    "laststate": ["static", "last"],
    "agg": ["static", "agg"],
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"]
}

datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
methods = encoding_dict[cls_encoding]
    
train_ratio = 0.8
n_splits = 3
random_state = 22

# create results directory
if not os.path.exists(os.path.join(params_dir)):
    os.makedirs(os.path.join(params_dir))
    
for dataset_name in datasets:
    
    # read the data
    dataset_manager = DatasetManager(dataset_name)
    data = dataset_manager.read_dataset()
    cls_encoder_args = {'case_id_col': dataset_manager.case_id_col, 
                        'static_cat_cols': dataset_manager.static_cat_cols,
                        'static_num_cols': dataset_manager.static_num_cols, 
                        'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                        'dynamic_num_cols': dataset_manager.dynamic_num_cols, 
                        'fillna': True}

    # determine min and max (truncated) prefix lengths
    min_prefix_length = 1
    if "traffic_fines" in dataset_name:
        max_prefix_length = 10
    elif "bpic2017" in dataset_name:
        max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
    else:
        max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))

    # split into training and test
    train, _ = dataset_manager.split_data_strict(data, train_ratio, split="temporal")
    
    # prepare chunks for CV
    dt_prefixes = []
    class_ratios = []
    for train_chunk, test_chunk in dataset_manager.get_stratified_split_generator(train, n_splits=n_splits):
        class_ratios.append(dataset_manager.get_class_ratio(train_chunk))
        # generate data where each prefix is a separate instance
        dt_prefixes.append(dataset_manager.generate_prefix_data(test_chunk, min_prefix_length, max_prefix_length))
    del train
        
    # set up search space
    if cls_method == "rf":
        space = {'max_features': hp.uniform('max_features', 0, 1)}
    elif cls_method == "xgboost":
        space = {'learning_rate': hp.uniform("learning_rate", 0, 1),
                 'subsample': hp.uniform("subsample", 0.5, 1),
                 'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
                 'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 1),
                 'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 6, 1))}
    elif cls_method == "logit":
        space = {'C': hp.uniform('C', -15, 15)}
    elif cls_method == "svm":
        space = {'C': hp.uniform('C', -15, 15),
                 'gamma': hp.uniform('gamma', -15, 15)}
    space['n_neighbors'] = scope.int(hp.quniform('n_neighbors', 2, 50, 1))

    # optimize parameters
    trial_nr = 1
    trials = Trials()
    fout_all = open(os.path.join(params_dir, "param_optim_all_trials_%s_%s_%s.csv" % (cls_method, dataset_name, method_name)), "w")
    if "prefix" in method_name:
        fout_all.write("%s;%s;%s;%s;%s;%s;%s;%s\n" % ("iter", "dataset", "cls", "method", "nr_events", "param", "value", "score"))   
    else:
        fout_all.write("%s;%s;%s;%s;%s;%s;%s\n" % ("iter", "dataset", "cls", "method", "param", "value", "score"))   
    best = fmin(create_and_evaluate_model, space, algo=tpe.suggest, max_evals=n_iter, trials=trials)
    fout_all.close()

    # write the best parameters
    best_params = hyperopt.space_eval(space, best)
    outfile = os.path.join(params_dir, "optimal_params_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name))
    # write to file
    with open(outfile, "wb") as fout:
        pickle.dump(best_params, fout)
