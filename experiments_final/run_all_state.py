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
from transformers.BooleanTransformer import BooleanTransformer

from bucketers.StateBasedBucketer import StateBasedBucketer

import dataset_confs

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

dataset_ref = argv[1]
method_name = argv[2]
results_dir = argv[3]

home_dir = ".."
optimal_params_filename = "optimal_params_state.pickle"

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
    "state_laststate": ["static", "laststate"],
    "state_agg": ["static", "agg"],
    "state_combined": ["static", "laststate", "agg"]}
    
datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
methods = methods_dict[method_name]

outfile = os.path.join(home_dir, results_dir, "final_results_%s_%s.csv"%(method_name, dataset_ref)) 

    
train_ratio = 0.8
rf_n_estimators = 500
random_state = 22
rf_max_features = None # assigned on the fly from the best params
n_min_cases_in_state = 30
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
        train = data[data[case_id_col].isin(train_ids)].sort_values(timestamp_col, ascending=True)
        test = data[~data[case_id_col].isin(train_ids)].sort_values(timestamp_col, ascending=True)
        del data
        del start_timestamps
        del train_ids
        
        train['case_length'] = train.groupby(case_id_col)[activity_col].transform(len)

        test_case_lengths = test.groupby(case_id_col).size()

        # generate prefix data (each possible prefix becomes a trace)
        print("Generating prefix data...")
        sys.stdout.flush()
        train_prefixes = train[train['case_length'] >= prefix_lengths[0]].groupby(case_id_col).head(prefix_lengths[0])
        for nr_events in prefix_lengths[1:]:
            tmp = train[train['case_length'] >= nr_events].groupby(case_id_col).head(nr_events)
            tmp[case_id_col] = tmp[case_id_col].apply(lambda x: "%s_%s"%(x, nr_events))
            train_prefixes = pd.concat([train_prefixes, tmp], axis=0)
        
        if dataset_name not in best_params or method_name not in best_params[dataset_name]:
            continue
        rf_max_features = best_params[dataset_name][method_name]['rf_max_features']

        # Assign states
        print("Assigning states...")
        start = time()
        boolean_encoder = BooleanTransformer(case_id_col=case_id_col, cat_cols=[activity_col], num_cols=[], fillna=fillna)
        data_bool = boolean_encoder.fit_transform(train_prefixes)
        bucketer = StateBasedBucketer()
        train_states = bucketer.fit_predict(data_bool)
        encoding_time_train += time() - start

        pipelines = {}
        hardcoded_predictions = {}

        # train and fit pipeline for each state
        for cl in range(bucketer.n_states):
            print("Fitting pipeline for state %s..."%cl)
            relevant_cases = data_bool[train_states == cl].index

            if len(relevant_cases) == 0:
                continue
            elif len(relevant_cases) < n_min_cases_in_state:
                dt_train_cluster = train_prefixes[train_prefixes[case_id_col].isin(relevant_cases)]
                train_y = [1 if label==pos_label else 0 for label in dt_train_cluster.groupby(case_id_col).first()[label_col]]
                hardcoded_predictions[cl] = np.mean(train_y)
            elif len(train_prefixes[train_prefixes[case_id_col].isin(relevant_cases)][label_col].unique()) < 2:
                hardcoded_predictions[cl] = train_prefixes[train_prefixes[case_id_col].isin(relevant_cases)][[label_col]].iloc[0]
            else:
                cls = RandomForestClassifier(n_estimators=rf_n_estimators, max_features=rf_max_features, random_state=random_state)
                feature_combiner = FeatureUnion([(method, init_encoder(method)) for method in methods])
                pipelines[cl] = Pipeline([('encoder', feature_combiner), ('cls', cls)])

                # fit pipeline
                dt_train_cluster = train_prefixes[train_prefixes[case_id_col].isin(relevant_cases)].sort_values(timestamp_col, ascending=True)
                train_y = dt_train_cluster.groupby(case_id_col).first()[label_col]

                start = time()
                pipelines[cl].fit(dt_train_cluster, train_y)
                pipeline_fit_time = time() - start

                train_encoding_fit_time = sum([el[1].fit_time for el in pipelines[cl].named_steps["encoder"].transformer_list])
                train_encoding_transform_time = sum([el[1].transform_time for el in pipelines[cl].named_steps["encoder"].transformer_list])
                encoding_time_train += train_encoding_fit_time
                encoding_time_train += train_encoding_transform_time
                cls_time_train += pipeline_fit_time - train_encoding_fit_time - train_encoding_transform_time

                
        # test separately for each prefix length
        for nr_events in prefix_lengths:
            print("Predicting for %s events..."%nr_events)

            # select only cases that are at least of length nr_events
            relevant_case_ids = test_case_lengths.index[test_case_lengths >= nr_events]
            
            if len(relevant_case_ids) == 0:
                break
            relevant_test = test[test[case_id_col].isin(relevant_case_ids)].sort_values(timestamp_col, ascending=True)
                
            del relevant_case_ids
                    
            start = time()
            # get predicted state for each test case
            test_data_cool = boolean_encoder.transform(relevant_test.groupby(case_id_col).head(nr_events))
            test_states = bucketer.predict(test_data_cool)
            online_time += time() - start

            # use appropriate classifier for each bucket of test cases
            preds = []
            test_y = []
            for cl in range(bucketer.n_states):
                current_cluster_case_ids = test_data_cool[test_states == cl].index
                current_cluster_grouped_test = relevant_test[relevant_test[case_id_col].isin(current_cluster_case_ids)].sort_values(timestamp_col, ascending=True).groupby(case_id_col, as_index=False)
                    
                if len(current_cluster_case_ids) == 0:
                    continue
                        
                elif cl not in pipelines:
                    if cl in hardcoded_predictions:
                        hardcoded_prediction = hardcoded_predictions[cl]
                    else:
                        hardcoded_prediction = 0.5
                        
                    current_cluster_preds = [hardcoded_prediction] * len(current_cluster_case_ids)
                    
                else:
                    # make predictions
                    preds_pos_label_idx = np.where(pipelines[cl].named_steps["cls"].classes_ == pos_label)[0][0] 
                    start = time()
                    current_cluster_preds = pipelines[cl].predict_proba(current_cluster_grouped_test.head(nr_events))[:,preds_pos_label_idx]
                    online_time += time() - start
                        
                preds.extend(current_cluster_preds)

                # extract actual label values
                current_cluster_test_y = [1 if label==pos_label else 0 for label in current_cluster_grouped_test.first()[label_col]]
                test_y.extend(current_cluster_test_y)
            #total_prediction_time = time() - start

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
