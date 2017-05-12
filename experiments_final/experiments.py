import pandas as pd
import numpy as np
import sys
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from time import time
import pickle
import os
from sys import argv

sys.path.append("..")
from transformers.StaticTransformer import StaticTransformer
from transformers.LastStateTransformer import LastStateTransformer
from transformers.AggregateTransformer import AggregateTransformer
from transformers.IndexBasedTransformer import IndexBasedTransformer
from transformers.BooleanTransformer import BooleanTransformer

from bucketers.StateBasedBucketer import StateBasedBucketer
from bucketers.PrefixLengthBucketer import PrefixLengthBucketer
from bucketers.NoBucketer import NoBucketer
from sklearn.cluster import KMeans

from DatasetManager import DatasetManager


dataset_ref = argv[1]
bucketing = argv[2]
bucket_encoding = argv[3]
cls_encoding = argv[4]
classifier = argv[5]
optimal_params_filename = argv[6]
results_dir = argv[7]

method_name = "%s_%s"%(bucketing, cls_encoding)

home_dir = ".."

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
    "freq": ["agg"],
    "bool": ["bool"],
    "combined": ["static", "last", "agg"]}
    
datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
methods = encoding_dict[cls_encoding]

outfile = os.path.join(home_dir, results_dir, "final_results_%s_%s_%s.csv"%(classifier, method_name, dataset_ref)) 

    
train_ratio = 0.8
random_state = 22
rf_n_estimators = 500 # only relevant if classifier == "rf"
fillna = True
n_min_cases_in_bucket = 30


def init_encoder(method, dataset_manager):
    
    if method == "static":
        return StaticTransformer(case_id_col=dataset_manager.case_id_col, cat_cols=dataset_manager.static_cat_cols, num_cols=dataset_manager.static_num_cols, fillna=fillna)
    
    elif method == "last":
        return LastStateTransformer(case_id_col=dataset_manager.case_id_col, cat_cols=dataset_manager.dynamic_cat_cols, num_cols=dataset_manager.dynamic_num_cols, fillna=fillna)
    
    elif method == "agg":
        return AggregateTransformer(case_id_col=dataset_manager.case_id_col, cat_cols=dataset_manager.dynamic_cat_cols, num_cols=dataset_manager.dynamic_num_cols, fillna=fillna)
    
    elif method == "bool":
        return BooleanTransformer(case_id_col=dataset_manager.case_id_col, cat_cols=dataset_manager.dynamic_cat_cols, num_cols=dataset_manager.dynamic_num_cols, fillna=fillna)
    
    else:
        print("Invalid encoder type")
        return None
    
    
def init_bucketer(method, dataset_manager):
    
    if method == "cluster":
        bucket_encoder = init_encoder(bucket_encoding, dataset_manager)
        bucket_encoder.cat_cols=[dataset_manager.activity_col]
        bucket_encoder.num_cols=[]
        n_clusters = best_params[dataset_name][method_name]['n_clusters']
        pipeline = Pipeline([('encoder', bucket_encoder), ('bucketer', KMeans(n_clusters, random_state=random_state))])
        return pipeline
    
    elif method == "state":
        bucket_encoder = init_encoder(bucket_encoding, dataset_manager)
        bucket_encoder.cat_cols=[dataset_manager.activity_col]
        bucket_encoder.num_cols=[]
        pipeline = Pipeline([('encoder', bucket_encoder), ('bucketer', StateBasedBucketer())])
        return pipeline
    
    elif method == "single":
        return NoBucketer(case_id_col=dataset_manager.case_id_col)
    
    elif method == "prefix":
        return PrefixLengthBucketer(case_id_col=dataset_manager.case_id_col)
    
    else:
        print("Invalid bucketer type")
        return None
    
    
def init_classifier(method):
    
    if method == "rf":
        return RandomForestClassifier(n_estimators=rf_n_estimators, max_features=best_params[dataset_name][method_name]['rf_max_features'], random_state=random_state)
                
    elif method == "gbm":
        return GradientBoostingClassifier(n_estimators=best_params[dataset_name][method_name][nr_events]['gbm_n_estimators'], max_features=best_params[dataset_name][method_name][nr_events]['gbm_max_features'], learning_rate=best_params[dataset_name][method_name][nr_events]['gbm_learning_rate'], random_state=random_state)
                
    else:
        print("Invalid classifier type")
        return None
                
    
    
##### MAIN PART ######    
with open(outfile, 'w') as fout:
    
    fout.write("%s;%s;%s;%s;%s\n"%("dataset", "method", "nr_events", "metric", "score"))
    
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
        
        # Bucketing prefixes based on control flow
        print("Bucketing prefixes...")
        bucketer = init_bucketer(bucketing, dataset_manager)
        bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)
            
        pipelines = {}
        hardcoded_predictions = {}

        # train and fit pipeline for each bucket
        for bucket in set(bucket_assignments_train):
            print("Fitting pipeline for bucket %s..."%bucket)
            relevant_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[bucket_assignments_train == bucket]
            dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes, relevant_cases_bucket) # one row per event
            train_y = dataset_manager.get_label(dt_train_bucket) # one row per case
            
            if len(relevant_cases_bucket) < n_min_cases_in_bucket:
                train_y = dataset_manager.get_label_numeric(dt_train_bucket)
                hardcoded_predictions[bucket] = np.mean(train_y)
            
            elif len(train_y.unique()) < 2:
                hardcoded_predictions[bucket] = 1 if str(train_y.iloc[0]) == dataset_manager.pos_label else 0
                
            else:
                feature_combiner = FeatureUnion([(method, init_encoder(method, dataset_manager)) for method in methods])
                pipelines[bucket] = Pipeline([('encoder', feature_combiner), ('cls', init_classifier(classifier))])

                # fit pipeline
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
            preds = []
            test_y = []
            for bucket in set(bucket_assignments_test):
                relevant_cases_bucket = dataset_manager.get_indexes(dt_test_prefixes)[bucket_assignments_test == bucket]
                dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_cases_bucket) # one row per event
                
                if len(relevant_cases_bucket) == 0:
                    continue
                    
                elif bucket not in pipelines:
                    
                    if bucket in hardcoded_predictions:
                        hardcoded_prediction = hardcoded_predictions[bucket]
                        
                    else:
                        hardcoded_prediction = dataset_manager.get_class_ratio(train)
                        
                    preds_bucket = [hardcoded_prediction] * len(relevant_cases_bucket)
                    
                elif len(pipelines[bucket].named_steps["cls"].classes_) == 1:
                    hardcoded_prediction = 1 if pipelines[bucket].named_steps["cls"].classes_[0] == dataset_manager.pos_label else 0
                    preds_bucket = [hardcoded_prediction] * len(relevant_cases_bucket)
                    
                else:
                    # make predictions
                    preds_pos_label_idx = np.where(pipelines[bucket].named_steps["cls"].classes_ == dataset_manager.pos_label)[0][0] 
                    preds_bucket = pipelines[bucket].predict_proba(dt_test_bucket)[:,preds_pos_label_idx]
                        
                preds.extend(preds_bucket)

                # extract actual label values
                test_y_bucket = dataset_manager.get_label_numeric(dt_test_bucket) # one row per case
                test_y.extend(test_y_bucket)

            if len(set(test_y)) < 2:
                auc = None
            else:
                auc = roc_auc_score(test_y, preds)
            prec, rec, fscore, _ = precision_recall_fscore_support(test_y, [0 if pred < 0.5 else 1 for pred in preds], average="binary")

            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "auc", auc))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "precision", prec))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "recall", rec))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "fscore", fscore))
            
        print("\n")
