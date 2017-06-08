import pandas as pd
import numpy as np
import sys
from time import time
from sys import argv
import os

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold

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

datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]

home_dir = ".."

if not os.path.exists(os.path.join(home_dir, results_dir)):
    os.makedirs(os.path.join(home_dir, results_dir))

methods_dict = {
    "state_laststate": ["static", "laststate"],
    "state_agg": ["static", "agg"],
    "state_combined": ["static", "laststate", "agg"]}


methods = methods_dict[method_name]

train_ratio = 0.8
rf_n_estimators = 500
rf_max_featuress = ["sqrt", 0.05, 0.1, 0.25, 0.5, 0.75]
random_state = 22
fillna = True
n_min_cases_in_state = 30


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
    
    
outfile = os.path.join(home_dir, results_dir, "cv_results_state_%s_%s.csv"%(dataset_ref, method_name))      
    
##### MAIN PART ######    
with open(outfile, 'w') as fout:
    
    fout.write("%s;%s;%s;%s;%s;%s\n"%("part", "dataset", "method", "rf_max_features", "metric", "score"))

    for dataset_name in datasets:
        print("Dataset %s..."%dataset_name)
        sys.stdout.flush()

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

        # consider prefix lengths until 90% of positive cases have finished
        min_prefix_length = 1
        prefix_lengths = list(range(min_prefix_length, min(20, int(np.ceil(data[data[label_col]==pos_label].groupby(case_id_col).size().quantile(0.90)))) + 1))

        # split into train and test using temporal split
        start_timestamps = data.groupby(case_id_col)[timestamp_col].min().reset_index()
        start_timestamps.sort_values(timestamp_col, ascending=1, inplace=True)
        train_ids = list(start_timestamps[case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[case_id_col].isin(train_ids)]
        del data
        del start_timestamps
        del train_ids

        grouped_train_firsts = train.groupby(case_id_col, as_index=False).first()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)

        part = 0
        for train_index, test_index in skf.split(grouped_train_firsts, grouped_train_firsts[label_col]):
            part += 1
            print("Starting chunk %s..."%part)
            sys.stdout.flush()

            # create train and validation data according to current fold
            current_train_names = grouped_train_firsts[case_id_col][train_index]
            train_chunk = train[train[case_id_col].isin(current_train_names)].sort_values(timestamp_col, ascending=True)
            test_chunk = train[~train[case_id_col].isin(current_train_names)].sort_values(timestamp_col, ascending=True)
            
            train_chunk['case_length'] = train_chunk.groupby(case_id_col)[activity_col].transform(len)
            test_chunk['case_length'] = test_chunk.groupby(case_id_col)[activity_col].transform(len)

            test_case_lengths = test_chunk.sort_values(timestamp_col, ascending=True).groupby(case_id_col).size()

            train_prefixes = train_chunk[train_chunk['case_length'] >= prefix_lengths[0]].sort_values(timestamp_col, ascending=True).groupby(case_id_col).head(prefix_lengths[0])
            train_prefixes[case_id_col] = train_prefixes[case_id_col].apply(lambda x: "%s_%s"%(x, prefix_lengths[0]))
            for nr_events in prefix_lengths[1:]:
                tmp = train_chunk[train_chunk['case_length'] >= nr_events].groupby(case_id_col).head(nr_events)
                tmp[case_id_col] = tmp[case_id_col].apply(lambda x: "%s_%s"%(x, nr_events))
                train_prefixes = pd.concat([train_prefixes, tmp], axis=0)

            # generate prefix data for testing (each possible prefix becomes a trace)
            print("Generating test prefix data...")
            sys.stdout.flush()
            start = time()
            test_prefixes = test_chunk[test_chunk['case_length'] >= prefix_lengths[0]].sort_values(timestamp_col, ascending=True).groupby(case_id_col).head(prefix_lengths[0])
            test_prefixes[case_id_col] = test_prefixes[case_id_col].apply(lambda x: "%s_%s"%(x, prefix_lengths[0]))
            for nr_events in prefix_lengths[1:]:
                tmp = test_chunk[test_chunk['case_length'] >= nr_events].groupby(case_id_col).head(nr_events)
                tmp[case_id_col] = tmp[case_id_col].apply(lambda x: "%s_%s"%(x, nr_events))
                test_prefixes = pd.concat([test_prefixes, tmp], axis=0)
            print(time() - start)
            sys.stdout.flush()
            
            test_y = [1 if label==pos_label else 0 for label in test_prefixes.groupby(case_id_col).first()[label_col]]

            # Assign states
            #boolean_encoder = BooleanTransformer(case_id_col=case_id_col, cat_cols=[activity_col], num_cols=[], fillna=fillna)
            boolean_encoder = LastStateTransformer(case_id_col=case_id_col, cat_cols=[activity_col], num_cols=[], fillna=fillna)
            data_bool = boolean_encoder.fit_transform(train_prefixes)
            bucketer = StateBasedBucketer()
            train_states = bucketer.fit_predict(data_bool)

            test_data_cool = boolean_encoder.transform(test_prefixes)
            test_states = bucketer.predict(test_data_cool)
        
            feature_combiners = {}
                    
            for rf_max_features in rf_max_featuress:
                print("RF max_features = %s"%(rf_max_features))
                sys.stdout.flush()
                
                preds = []
                test_y = []
                for cl in range(bucketer.n_states):
                    print("Fitting pipeline for state %s..."%cl)
                    sys.stdout.flush()

                    relevant_train_cases = data_bool[train_states == cl].index
                    relevant_test_cases = test_data_cool[test_states == cl].index

                    if len(relevant_test_cases) == 0:
                        continue
                        
                    elif len(relevant_train_cases) < n_min_cases_in_state:
                        if len(relevant_train_cases) == 0:
                            hardcoded_prediction = 0.5
                        else:
                            dt_train_cluster = train_prefixes[train_prefixes[case_id_col].isin(relevant_train_cases)]
                            train_y = [1 if label==pos_label else 0 for label in dt_train_cluster.groupby(case_id_col).first()[label_col]]
                            hardcoded_prediction = np.mean(train_y)
                        
                        current_cluster_preds = [hardcoded_prediction] * len(relevant_test_cases)
                        dt_test_cluster = test_prefixes[test_prefixes[case_id_col].isin(relevant_test_cases)].sort_values(timestamp_col, ascending=True)
                    
                    else:
                        dt_train_cluster = train_prefixes[train_prefixes[case_id_col].isin(relevant_train_cases)].sort_values(timestamp_col, ascending=True)
                        train_y = dt_train_cluster.groupby(case_id_col).first()[label_col]
                        del relevant_train_cases

                        dt_test_cluster = test_prefixes[test_prefixes[case_id_col].isin(relevant_test_cases)].sort_values(timestamp_col, ascending=True)

                        #### ENCODE DATA ####
                        if cl not in feature_combiners:
                            start = time()
                            feature_combiners[cl] = FeatureUnion([(method, init_encoder(method)) for method in methods])
                            dt_train = feature_combiners[cl].fit_transform(dt_train_cluster)
                            encode_time_train = time() - start
                        else:
                            dt_train = feature_combiners[cl].transform(dt_train_cluster)

                        start = time()
                        dt_test = feature_combiners[cl].transform(dt_test_cluster)
                        encode_time_test = time() - start

                        #### FIT CLASSIFIER ####
                        cls = RandomForestClassifier(n_estimators=rf_n_estimators, max_features=rf_max_features, random_state=random_state)
                        cls.fit(dt_train, train_y)

                        #### PREDICT ####
                        start = time()
                        if len(train_y.unique()) == 1:
                            hardcoded_prediction = 1 if train_y[0] == pos_label else 0
                            current_cluster_preds = [hardcoded_prediction] * len(relevant_test_cases)
                        else:
                            # make predictions
                            preds_pos_label_idx = np.where(cls.classes_ == pos_label)[0][0] 
                            current_cluster_preds = cls.predict_proba(dt_test)[:,preds_pos_label_idx]
                        prediction_time = time() - start
                    
                                
                    preds.extend(current_cluster_preds)

                    # extract actual label values
                    current_cluster_test_y = [1 if label==pos_label else 0 for label in dt_test_cluster.groupby(case_id_col).first()[label_col]]
                    test_y.extend(current_cluster_test_y)

                    #### EVALUATE OVER ALL CLUSTERS ####    
                    if len(set(test_y)) < 2:
                        auc = None
                    else:
                        auc = roc_auc_score(test_y, preds)
                    prec, rec, fscore, _ = precision_recall_fscore_support(test_y, [0 if pred < 0.5 else 1 for pred in preds], average="binary")

                    fout.write("%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, "auc", auc))
                    fout.write("%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, "precision", prec))
                    fout.write("%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, "recall", rec))
                    fout.write("%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, "fscore", fscore))

            print("\n")


