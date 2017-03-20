import pandas as pd
import numpy as np
import sys
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
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
from transformers.HMMDiscriminativeTransformer import HMMDiscriminativeTransformer
from transformers.HMMGenerativeTransformer import HMMGenerativeTransformer

import dataset_confs

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

home_dir = ".."
results_dir = "final_results2"
optimal_params_filename = "optimal_params.pickle"

if not os.path.exists(os.path.join(home_dir, results_dir)):
    os.makedirs(os.path.join(home_dir, results_dir))

dataset_ref = argv[1]
method_name = argv[2]

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
    "index_laststate": ["static", "laststate"],
    "index_agg": ["static", "agg"]}
    
datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
methods = methods_dict[method_name]

outfile = os.path.join(home_dir, results_dir, "final_results_%s_%s.csv"%(method_name, dataset_ref)) 

    
train_ratio = 0.8
hmm_min_seq_length = 2
hmm_max_seq_length = None
hmm_n_iter = 30
hmm_n_states = 6
rf_n_estimators = 500
random_state = 22
rf_max_features = None # assigned on the fly from the best params
fillna = True



def init_encoder(method):
    
    if method == "static":
        return StaticTransformer(case_id_col=case_id_col, cat_cols=static_cat_cols, num_cols=static_num_cols, fillna=fillna)
    
    elif method == "laststate":
        return LastStateTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, fillna=fillna)
    
    elif method == "agg":
        return AggregateTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, fillna=fillna)
    
    elif method == "hmm_disc":
        hmm_disc_encoder = HMMDiscriminativeTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols,
                                                        num_cols=dynamic_num_cols, n_states=hmm_n_states, label_col=label_col,
                                                        pos_label=pos_label, min_seq_length=hmm_min_seq_length,
                                                        max_seq_length=hmm_max_seq_length, random_state=random_state,
                                                        n_iter=hmm_n_iter, fillna=fillna)
        hmm_disc_encoder.fit(train.sort_values(timestamp_col, ascending=True))
        return hmm_disc_encoder
    
    elif method == "hmm_gen":
        hmm_gen_encoder = HMMGenerativeTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols,
                                                   num_cols=dynamic_num_cols, n_states=hmm_n_states,
                                                   min_seq_length=hmm_min_seq_length, max_seq_length=hmm_max_seq_length,
                                                   random_state=random_state, n_iter=hmm_n_iter, fillna=fillna)
        hmm_gen_encoder.fit(train.sort_values(timestamp_col, ascending=True))
        return hmm_gen_encoder
    
    else:
        print("Invalid encoder type")
        return None
    
    
##### MAIN PART ######    
with open(outfile, 'w') as fout:
    
    fout.write("%s;%s;%s;%s;%s\n"%("dataset", "method", "nr_events", "metric", "score"))
    
    for dataset_name in datasets:
        
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
        
        data_filepath = os.path.join(home_dir, dataset_confs.filename[dataset_name])

        dtypes = {col:"object" for col in dynamic_cat_cols+static_cat_cols+[case_id_col, label_col, timestamp_col]}
        for col in dynamic_num_cols + static_num_cols:
            dtypes[col] = "float"
        
        data = pd.read_csv(data_filepath, sep=";", dtype=dtypes)
        data[timestamp_col] = pd.to_datetime(data[timestamp_col])
        
        # maximum prefix length considered can't be larger than the 75th quantile
        min_prefix_length = 1 if "hmm_disc" not in methods else 2
        prefix_lengths = list(range(min_prefix_length, min(20, int(np.ceil(data.groupby(case_id_col).size().quantile(0.75)))) + 1))

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

        grouped_train = train.sort_values(timestamp_col, ascending=True).groupby(case_id_col)
        grouped_test = test.sort_values(timestamp_col, ascending=True).groupby(case_id_col)
        
        # generate prefix data (each possible prefix becomes a trace)
        print("Generating prefix data...")
        train_prefixes = grouped_train.head(prefix_lengths[0])
        for nr_events in prefix_lengths[1:]:
            tmp = grouped_train.head(nr_events)
            tmp[case_id_col] = tmp[case_id_col].apply(lambda x: "%s_%s"%(x, nr_events))
            train_prefixes = pd.concat([train_prefixes, tmp], axis=0)
        del grouped_train 
        
        print("Generating test prefix data...")
        test_prefixes = grouped_test.head(prefix_lengths[0])
        for nr_events in prefix_lengths[1:]:
            tmp = grouped_test.head(nr_events)
            tmp[case_id_col] = tmp[case_id_col].apply(lambda x: "%s_%s"%(x, nr_events))
            test_prefixes = pd.concat([test_prefixes, tmp], axis=0)
        del grouped_test 
        
        train_case_lengths = train_prefixes.sort_values(timestamp_col, ascending=True).groupby(case_id_col).size()
        test_case_lengths = test_prefixes.sort_values(timestamp_col, ascending=True).groupby(case_id_col).size()
        
        rf_max_features = best_params[dataset_name][method_name]['rf_max_features']

        # test separately for each prefix length
        for nr_events in prefix_lengths:
            
            #### SELECT RELEVANT CASES ####
            print("Selecting relevant cases for %s events..."%nr_events)
            sys.stdout.flush()
            relevant_train_case_names = train_case_lengths[train_case_lengths == nr_events].index
            relevant_test_case_names = test_case_lengths[test_case_lengths == nr_events].index

            if len(relevant_test_case_names) == 0:
                continue
            elif len(relevant_train_case_names) == 0:
                preds = [0.5] * len(relevant_test_case_names)
                continue
                
            
            #### FIT PIPELINE ####
            print("Fitting pipeline for %s events..."%nr_events)
            sys.stdout.flush()
            cls = RandomForestClassifier(n_estimators=rf_n_estimators, max_features=rf_max_features, random_state=random_state)
            feature_combiner = FeatureUnion([(method, init_encoder(method)) for method in methods])
            pipeline = Pipeline([('encoder', feature_combiner), ('cls', cls)])
            
            start = time()
            dt_train = train_prefixes[train_prefixes[case_id_col].isin(relevant_train_case_names)].sort_values(timestamp_col, ascending=True)
            del relevant_train_case_names
            train_y = dt_train.groupby(case_id_col).first()[label_col]
            pipeline.fit(dt_train, train_y)
            fit_time = time() - start
            
            del dt_train
            del train_y
            
            #### PREDICT ####
            print("Predicting for %s events..."%nr_events)
            sys.stdout.flush()
            dt_test = test_prefixes[test_prefixes[case_id_col].isin(relevant_test_case_names)].sort_values(timestamp_col, ascending=True)
            del relevant_test_case_names
            test_y = [1 if label==pos_label else 0 for label in dt_test.groupby(case_id_col).first()[label_col]]
                
            start = time()
            if len(pipeline.named_steps["cls"].classes_) == 1:
                hardcoded_prediction = 1 if pipeline.named_steps["cls"].classes_[0] == pos_label else 0
                preds = [hardcoded_prediction] * len(current_cluster_case_ids)
            else:
                # make predictions
                preds_pos_label_idx = np.where(pipeline.named_steps["cls"].classes_ == pos_label)[0][0] 
                preds = pipeline.predict_proba(dt_test)[:,preds_pos_label_idx]
            prediction_time = time() - start

            if len(set(test_y)) < 2:
                auc = None
            else:
                auc = roc_auc_score(test_y, preds)
            prec, rec, fscore, _ = precision_recall_fscore_support(test_y, [0 if pred < 0.5 else 1 for pred in preds], average="binary")

            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "auc", auc))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "precision", prec))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "recall", rec))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "fscore", fscore))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "prediction_time", prediction_time))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "fit_time", fit_time))
            
            print("\n")
