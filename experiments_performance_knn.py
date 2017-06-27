import pandas as pd
import numpy as np
import sys
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.neighbors import NearestNeighbors
from time import time
import pickle
import os
from sys import argv

import EncoderFactory
import BucketFactory
import ClassifierFactory
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
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"]}
    
datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
methods = encoding_dict[cls_encoding]

outfile = os.path.join(home_dir, results_dir, "performance_results_%s_%s_%s.csv"%(cls_method, method_name, dataset_ref)) 

    
train_ratio = 0.8
random_state = 22
fillna = True
n_min_cases_in_bucket = 30
n_iter = 5
    
    
##### MAIN PART ######    
with open(outfile, 'w') as fout:
    
    fout.write("%s;%s;%s;%s;%s\n"%("dataset", "method", "iter", "metric", "score"))
    
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


        # extract arguments
        bucketer_args = {'encoding_method':bucket_encoding, 
                         'case_id_col':dataset_manager.case_id_col, 
                         'cat_cols':[dataset_manager.activity_col], 
                         'num_cols':[], 
                         'n_clusters':None, 
                         'random_state':random_state}
        if bucket_method == "cluster":
            bucketer_args['n_clusters'] = best_params[dataset_name][method_name][cls_method]['n_clusters']
        
        cls_encoder_args = {'case_id_col':dataset_manager.case_id_col, 
                            'static_cat_cols':dataset_manager.static_cat_cols,
                            'static_num_cols':dataset_manager.static_num_cols, 
                            'dynamic_cat_cols':dataset_manager.dynamic_cat_cols,
                            'dynamic_num_cols':dataset_manager.dynamic_num_cols, 
                            'fillna':fillna}
        
        
        offline_total_times = []
        online_event_times = []
        
        for ii in range(n_iter):
            
            start_prefix_generation = time()
            # create prefix logs
            dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
            dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)
            prefix_generation_time = time() - start_prefix_generation
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, ii, "prefix_generation_time", prefix_generation_time))
            

            n_neighbors = best_params[dataset_name][method_name][cls_method]['n_neighbors']
        
            # extract arguments
            knn_encoder_args = {'case_id_col':dataset_manager.case_id_col, 
                        'static_cat_cols':[],
                        'static_num_cols':[], 
                        'dynamic_cat_cols':[dataset_manager.activity_col],
                        'dynamic_num_cols':[], 
                        'fillna':fillna}

            cls_encoder_args = {'case_id_col':dataset_manager.case_id_col, 
                                'static_cat_cols':dataset_manager.static_cat_cols,
                                'static_num_cols':dataset_manager.static_num_cols, 
                                'dynamic_cat_cols':dataset_manager.dynamic_cat_cols,
                                'dynamic_num_cols':dataset_manager.dynamic_num_cols, 
                                'fillna':fillna}

            cls_args = {'max_features': best_params[dataset_name][method_name][cls_method]['max_features'],
                        'n_estimators': 500,
                        'random_state': random_state,
                        'min_cases_for_training': n_min_cases_in_bucket}
        
            start = time()
            # initiate the KNN model
            cf_encoder = EncoderFactory.get_encoder(bucket_encoding, **knn_encoder_args)
            encoded_train = cf_encoder.fit_transform(dt_train_prefixes)
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(encoded_train)
            
            offline_total_time = time() - start
            offline_total_times.append(offline_total_time)
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, ii, "offline_time_total", offline_total_time))
            
            current_online_event_times = []
        
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
                # get nearest neighbors for each test case
                encoded_test = cf_encoder.fit_transform(dt_test_nr_events)
                _, indices = nbrs.kneighbors(encoded_test)

                # use appropriate classifier for each bucket of test cases
                # for evaluation, collect predictions from different buckets together
                for i in range(len(encoded_test)):
                    
                    start = time()

                    # retrieve nearest neighbors from training set
                    knn_idxs = indices[i]
                    relevant_cases_bucket = encoded_train.iloc[knn_idxs].index
                    dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes, relevant_cases_bucket) # one row per event
                    train_y = dataset_manager.get_label_numeric(dt_train_bucket)

                    feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])
                    pipeline = Pipeline([('encoder', feature_combiner), ('cls', ClassifierFactory.get_classifier(cls_method, **cls_args))])

                    # fit the classifier based on nearest neighbors
                    pipeline.fit(dt_train_bucket, train_y)

                    # select current test case
                    relevant_test_case = [encoded_test.index[i]]
                    dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_nr_events, relevant_test_case)

                    # predict
                    pipeline.predict_proba(dt_test_bucket)
                    
                    pipeline_pred_time = time() - start
                    current_online_event_times.append(pipeline_pred_time / nr_events)
            

            online_event_times.extend(current_online_event_times)
            
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, ii, "online_time_avg", np.mean(current_online_event_times)))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, ii, "online_time_std", np.std(current_online_event_times)))
            
        fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, -1, "online_time_avg", np.mean(online_event_times)))
        fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, -1, "online_time_std", np.std(online_event_times)))
        fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, -1, "offline_time_total_avg", np.mean(offline_total_times)))
        fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, -1, "offline_time_total_std", np.std(offline_total_times)))

