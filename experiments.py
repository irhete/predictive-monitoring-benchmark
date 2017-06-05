import pandas as pd
import numpy as np
import sys
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline, FeatureUnion
from time import time
import pickle
import os
from sys import argv

from EncoderFactory import EncoderFactory
from BucketFactory import BucketFactory
from ClassifierFactory import ClassifierFactory
from DatasetManager import DatasetManager

dataset_ref = argv[1]
bucket_encoding = argv[2]
bucket_method = argv[3]
cls_encoding = argv[4]
cls_method = argv[5]
optimal_params_filename = argv[6]
results_dir = argv[7]

method_name = "%s_%s"%(bucket_method, cls_encoding)

home_dir = ""

if not os.path.exists(os.path.join(home_dir, results_dir)):
    os.makedirs(os.path.join(home_dir, results_dir))

with open(os.path.join(home_dir, optimal_params_filename), "rb") as fin:
    best_params = pickle.load(fin)

dataset_ref_to_datasets = {
    "bpic2011": ["bpic2011_f%s"%formula for formula in range(1,5)],
    "bpic2015": ["bpic2015_%s_f2"%(municipality) for municipality in range(1,6)],
    "insurance": ["insurance_activity", "insurance_followup"],
    "bpic2017": ["bpic2017"]
}

encoding_dict = {
    "laststate": ["static", "last"],
    "agg": ["static", "agg"],
    "combined": ["static", "last", "agg"]}
    
datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
methods = encoding_dict[cls_encoding]

outfile = os.path.join(home_dir, results_dir, "final_results_%s_%s_%s.csv"%(cls_method, method_name, dataset_ref)) 

    
train_ratio = 0.8
random_state = 22
fillna = True
n_min_cases_in_bucket = 30
    
    
##### MAIN PART ######    
with open(outfile, 'w') as fout:
    
    fout.write("%s;%s;%s;%s;%s;%s\n"%("dataset", "method", "cls", "nr_events", "metric", "score"))
    
    for dataset_name in datasets:
        
        dataset_manager = DatasetManager(dataset_name)
        
        # read the data
        data = dataset_manager.read_dataset()
        
        # split data into train and test
        train, test = dataset_manager.split_data(data, train_ratio)
        
        # consider prefix lengths until 90% of positive cases have finished
        min_prefix_length = 1
        max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
        del data

        # create prefix logs
        dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
        dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)

        print(dt_train_prefixes.shape)
        print(dt_test_prefixes.shape)
        
        # extract arguments
        bucketer_args = {'encoding_method':bucket_encoding, 
                         'case_id_col':dataset_manager.case_id_col, 
                         'cat_cols':[dataset_manager.activity_col], 
                         'num_cols':[], 
                         'n_clusters':best_params[dataset_name][method_name][cls_method]['n_clusters'], 
                         'random_state':random_state}
        
        cls_encoder_args = {'case_id_col':dataset_manager.case_id_col, 
                            'static_cat_cols':dataset_manager.static_cat_cols,
                            'static_num_cols':dataset_manager.static_num_cols, 
                            'dynamic_cat_cols':dataset_manager.dynamic_cat_cols,
                            'dynamic_num_cols':dataset_manager.dynamic_num_cols, 
                            'fillna':fillna}
        
        cls_args = {k:v for k,v in best_params[dataset_name][method_name][cls_method].items() if k != 'n_clusters'}
        cls_args['random_state'] = random_state
        cls_args['min_cases_for_training'] = n_min_cases_in_bucket
        
        # Bucketing prefixes based on control flow
        print("Bucketing prefixes...")
        bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)
        bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)
            
        pipelines = {}

        # train and fit pipeline for each bucket
        for bucket in set(bucket_assignments_train):
            print("Fitting pipeline for bucket %s..."%bucket)
            relevant_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[bucket_assignments_train == bucket]
            dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes, relevant_cases_bucket) # one row per event
            train_y = dataset_manager.get_label_numeric(dt_train_bucket)
            
            feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])
            pipelines[bucket] = Pipeline([('encoder', feature_combiner), ('cls', ClassifierFactory.get_classifier(cls_method, **cls_args))])
            
            pipelines[bucket].fit(dt_train_bucket, train_y)
            
            
        
        prefix_lengths_test = dt_test_prefixes.groupby(dataset_manager.case_id_col).size()
        
        # test separately for each prefix length
        for nr_events in range(min_prefix_length, max_prefix_length+1):
            print("Predicting for %s events..."%nr_events)

            # select only cases that are at least of length nr_events
            relevant_cases_nr_events = prefix_lengths_test[prefix_lengths_test == nr_events].index
            
            if len(relevant_cases_nr_events) == 0:
                break
                
            dt_test_nr_events = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_cases_nr_events)
            del relevant_cases_nr_events
                    
            start = time()
            # get predicted cluster for each test case
            bucket_assignments_test = bucketer.predict(dt_test_nr_events)

            # use appropriate classifier for each bucket of test cases
            # for evaluation, collect predictions from different buckets together
            preds = []
            test_y = []
            for bucket in set(bucket_assignments_test):
                relevant_cases_bucket = dataset_manager.get_indexes(dt_test_prefixes)[bucket_assignments_test == bucket]
                dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_cases_bucket) # one row per event
                
                if len(relevant_cases_bucket) == 0:
                    continue
                    
                elif bucket not in pipelines:
                    # use the general class ratio (in training set) as prediction 
                    preds_bucket = [dataset_manager.get_class_ratio(train)] * len(relevant_cases_bucket)
                    
                else:
                    # make actual predictions
                    preds_bucket = pipelines[bucket].predict_proba(dt_test_bucket)
                        
                preds.extend(preds_bucket)

                # extract actual label values
                test_y_bucket = dataset_manager.get_label_numeric(dt_test_bucket) # one row per case
                test_y.extend(test_y_bucket)

            if len(set(test_y)) < 2:
                auc = None
            else:
                auc = roc_auc_score(test_y, preds)
            prec, rec, fscore, _ = precision_recall_fscore_support(test_y, [0 if pred < 0.5 else 1 for pred in preds], average="binary")

            fout.write("%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, nr_events, "auc", auc))
            fout.write("%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, nr_events, "precision", prec))
            fout.write("%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, nr_events, "recall", rec))
            fout.write("%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, nr_events, "fscore", fscore))
            
        print("\n")
