import EncoderFactory
from DatasetManager import DatasetManager
import BucketFactory

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

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


dataset_ref = argv[1]
params_dir = argv[2]
results_dir = argv[3]
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
random_state = 22

# create results directory
if not os.path.exists(os.path.join(params_dir)):
    os.makedirs(os.path.join(params_dir))
    
for dataset_name in datasets:
    
    count_trivial = 0
    count_normal = 0
    
    # load optimal params
    optimal_params_filename = os.path.join(params_dir, "optimal_params_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name))
    if not os.path.isfile(optimal_params_filename) or os.path.getsize(optimal_params_filename) <= 0:
        continue
        
    with open(optimal_params_filename, "rb") as fin:
        args = pickle.load(fin)
    
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
    train, test = dataset_manager.split_data_strict(data, train_ratio, split="temporal")
    
    dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)
    dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)

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
    if "n_neighbors" in args:
        n_neighbors = int(args["n_neighbors"])
    else:
        n_neighbors = 50
    bucketer = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(encoded_train)

    for _, dt_test_bucket in dt_test_prefixes.groupby(dataset_manager.case_id_col):
        encoded_case = bucket_encoder.fit_transform(dt_test_bucket)
        _, knn_idxs = bucketer.kneighbors(encoded_case)
        knn_idxs = knn_idxs[0]
                
        relevant_cases_bucket = encoded_train.iloc[knn_idxs].index
        dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes, relevant_cases_bucket) # one row per event
        train_y = dataset_manager.get_label_numeric(dt_train_bucket)

        if len(set(train_y)) < 2:
            count_trivial += 1
        else:
            count_normal += 1
    print(dataset_name, count_trivial, count_normal)