import pandas as pd
import numpy as np
import sys
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cluster import KMeans
from time import time
import pickle
import os
from sys import argv

sys.path.append("..")
from transformers.StaticTransformer import StaticTransformer
from transformers.LastStateTransformer import LastStateTransformer
from transformers.AggregateTransformer import AggregateTransformer
from transformers.IndexBasedTransformer import IndexBasedTransformer

import dataset_confs

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

dataset_ref = argv[1]
method_name = argv[2]
results_dir = argv[3]
classifier = argv[4]

if classifier == "gbm":
    optimal_params_filename = "optimal_params_gbm.pickle"
elif classfier == "rf":
    optimal_params_filename = "optimal_params4.pickle"
    rf_n_estimators = 500
    
    
home_dir = ".."

if not os.path.exists(os.path.join(home_dir, results_dir)):
    os.makedirs(os.path.join(home_dir, results_dir))

with open(os.path.join(home_dir, optimal_params_filename), "rb") as fin:
    best_params = pickle.load(fin)

dataset_ref_to_datasets = {
    "bpic2011": ["bpic2011_f%s"%formula for formula in range(1,5)],
    #"bpic2015_f1": ["bpic2015_%s_f1"%(municipality) for municipality in range(1,6)],
    "bpic2015": ["bpic2015_%s_f2"%(municipality) for municipality in range(1,6)],
    "insurance": ["insurance_activity", "insurance_followup"],
    "sepsis_cases": ["sepsis_cases"],
    "traffic_fines": ["traffic_fines"],
    "siae": ["siae"],
    "bpic2017": ["bpic2017"]
}
dataset_ref_to_datasets["small_logs"] = dataset_ref_to_datasets["bpic2011"] + dataset_ref_to_datasets["bpic2015"] + dataset_ref_to_datasets["insurance"] + dataset_ref_to_datasets["sepsis_cases"]


methods_dict = {
    "index": ["static", "index"], # slow here. faster version in run_index_orig.py
    "index_laststate": ["static", "laststate"],
    "index_agg": ["static", "agg"],
    "index_hmm": ["static", "hmm_disc"]}
    
datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
methods = methods_dict[method_name]

outfile = os.path.join(home_dir, results_dir, "final_results_%s_%s_%s.csv"%(classifier, method_name, dataset_ref)) 

    
train_ratio = 0.8
random_state = 22
fillna = True



def init_encoder(method):
    
    if method == "static":
        return StaticTransformer(case_id_col=case_id_col, cat_cols=static_cat_cols, num_cols=static_num_cols, fillna=fillna)
    
    elif method == "laststate":
        return LastStateTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, fillna=fillna)
    
    elif method == "agg":
        return AggregateTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, fillna=fillna)
    
    else:
        print("Invalid encoder type")
        return None
    
    
##### MAIN PART ######    
with open(outfile, 'w') as fout:
    
    fout.write("%s;%s;%s;%s;%s\n"%("dataset", "method", "nr_events", "metric", "score"))
    
    for dataset_name in datasets:
        
        encoding_time_train = 0
        cls_time_train = 0
        online_time = 0
        
        # read dataset settings
        case_id_col = dataset_confs.case_id_col[dataset_name]
        activity_col = dataset_confs.activity_col[dataset_name]
        timestamp_col = dataset_confs.timestamp_col[dataset_name]
        label_col = dataset_confs.label_col[dataset_name]
        pos_label = dataset_confs.pos_label[dataset_name]

        dynamic_cat_cols = dataset_confs.dynamic_cat_cols[dataset_name]
        static_cat_cols = dataset_confs.static_cat_cols[dataset_name]
        dynamic_num_cols = dataset_confs.dynamic_num_cols[dataset_name]
        static_num_cols = dataset_confs.static_num_cols[dataset_name]
        
        data_filepath = dataset_confs.filename[dataset_name]

        dtypes = {col:"object" for col in dynamic_cat_cols+static_cat_cols+[case_id_col, label_col, timestamp_col]}
        for col in dynamic_num_cols + static_num_cols:
            dtypes[col] = "float"
        
        data = pd.read_csv(data_filepath, sep=";", dtype=dtypes)
        data[timestamp_col] = pd.to_datetime(data[timestamp_col])
        
        # consider prefix lengths until 90% of positive cases have finished
        min_prefix_length = 1
        prefix_lengths = list(range(min_prefix_length, min(20, int(np.ceil(data[data[label_col]==pos_label].groupby(case_id_col).size().quantile(0.90)))) + 1))

        # split into train and test using temporal split
        grouped = data.groupby(case_id_col)
        start_timestamps = grouped[timestamp_col].min().reset_index()
        start_timestamps.sort_values(timestamp_col, ascending=1, inplace=True)
        train_ids = list(start_timestamps[case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[case_id_col].isin(train_ids)]
        test = data[~data[case_id_col].isin(train_ids)]
        del data
        del start_timestamps
        del train_ids
        
        test_case_lengths = test.sort_values(timestamp_col, ascending=True).groupby(case_id_col).size()
        
        # test separately for each prefix length
        for nr_events in prefix_lengths:
            if dataset_name not in best_params or method_name not in best_params[dataset_name] or nr_events not in best_params[dataset_name][method_name]:
                continue

            #### SELECT RELEVANT CASES ####
            print("Selecting relevant cases for %s events..."%nr_events)
            sys.stdout.flush()
            
            # discard traces shorter than current prefix length from the test set
            relevant_test_case_names = test_case_lengths.index[test_case_lengths >= nr_events]

            if len(relevant_test_case_names) == 0:
                continue
                
            
            #### FIT PIPELINE ####
            print("Fitting pipeline for %s events..."%nr_events)
            sys.stdout.flush()
            
            if classifier == "rf":
                cls = RandomForestClassifier(n_estimators=rf_n_estimators, max_features=best_params[dataset_name][method_name][nr_events]['rf_max_features'], random_state=random_state)

            elif classifier == "gbm":
                cls = GradientBoostingClassifier(n_estimators=best_params[dataset_name][method_name][nr_events]['gbm_n_estimators'], max_features=best_params[dataset_name][method_name][nr_events]['gbm_max_features'], learning_rate=best_params[dataset_name][method_name][nr_events]['gbm_learning_rate'], random_state=random_state)

            else:
                print("Classifier unknown")
                break             
            
            feature_combiner = FeatureUnion([(method, init_encoder(method)) for method in methods])
            pipeline = Pipeline([('encoder', feature_combiner), ('cls', cls)])
            
            # training set contains as many traces as are in the original log. shorter ones contain imputed values. 
            dt_train = train.sort_values(timestamp_col, ascending=1).groupby(case_id_col).head(nr_events)
            train_y = dt_train.groupby(case_id_col).first()[label_col]
            
            start = time()
            pipeline.fit(dt_train, train_y)
            pipeline_fit_time = time() - start
            
            train_encoding_fit_time = sum([el[1].fit_time for el in pipeline.named_steps["encoder"].transformer_list])
            train_encoding_transform_time = sum([el[1].transform_time for el in pipeline.named_steps["encoder"].transformer_list])
            encoding_time_train += train_encoding_fit_time
            encoding_time_train += train_encoding_transform_time
            cls_time_train += pipeline_fit_time - train_encoding_fit_time - train_encoding_transform_time
            
            del dt_train
            del train_y
            
            #### PREDICT ####
            print("Predicting for %s events..."%nr_events)
            sys.stdout.flush()
            dt_test = test[test[case_id_col].isin(relevant_test_case_names)].sort_values(timestamp_col, ascending=1).groupby(case_id_col).head(nr_events)
            test_y = [1 if label==pos_label else 0 for label in dt_test.groupby(case_id_col).first()[label_col]]
                
            start = time()
            if len(pipeline.named_steps["cls"].classes_) == 1:
                hardcoded_prediction = 1 if pipeline.named_steps["cls"].classes_[0] == pos_label else 0
                preds = [hardcoded_prediction] * len(relevant_test_case_names)
            else:
                # make predictions
                preds_pos_label_idx = np.where(pipeline.named_steps["cls"].classes_ == pos_label)[0][0] 
                preds = pipeline.predict_proba(dt_test)[:,preds_pos_label_idx]
            online_time += time() - start

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
            
        fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, 0, "encoding_time_train", encoding_time_train))
        fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, 0, "cls_time_train", cls_time_train))
        fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, 0, "online_time", online_time))
            
            
