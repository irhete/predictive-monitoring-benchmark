import pandas as pd
import numpy as np
import sys
from time import time
from sys import argv
import os

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import StratifiedKFold

sys.path.append("..")
from transformers.StaticTransformer import StaticTransformer
from transformers.LastStateTransformer import LastStateTransformer
from transformers.PreviousStateTransformer import PreviousStateTransformer
from transformers.AggregateTransformer import AggregateTransformer
from transformers.HMMDiscriminativeTransformer import HMMDiscriminativeTransformer
from transformers.IndexBasedTransformer import IndexBasedTransformer
import dataset_confs

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


dataset_ref = argv[1]
method_name = argv[2]

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

home_dir = ".."
results_dir = "cv_results2"

methods_dict = {
    "index_index": ["static", "index"],
    "index_laststate": ["static", "laststate"],
    "index_agg": ["static", "agg"],
    "index_hmm": ["static", "hmm_disc"]}
    
datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
methods = methods_dict[method_name]

train_ratio = 0.8
hmm_min_seq_length = 2
hmm_max_seq_length = None
hmm_n_iter = 50
hmm_n_statess = [0] if "hmm_disc" not in methods else [1, 2, 3, 4, 6, 8, 10]
rf_n_estimators = 500
rf_max_featuress = ["sqrt", 0.05, 0.1, 0.25, 0.5, 0.75]
random_state = 22
fillna = True

def init_encoder(method):
    
    if method == "static":
        return StaticTransformer(case_id_col=case_id_col, cat_cols=static_cat_cols, num_cols=static_num_cols, fillna=fillna)
    
    elif method == "laststate":
        return LastStateTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, fillna=fillna)
    
    elif method == "prevstate":
        return PreviousStateTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, fillna=fillna)
    
    elif method == "agg":
        return AggregateTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, fillna=fillna)
    elif method == "index":
        return IndexBasedTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, fillna=fillna)
    elif method == "hmm_disc":
        hmm_disc_encoder = HMMDiscriminativeTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols,
                                                        num_cols=dynamic_num_cols, n_states=hmm_n_states, label_col=label_col,
                                                        pos_label=pos_label, min_seq_length=hmm_min_seq_length,
                                                        max_seq_length=hmm_max_seq_length, random_state=random_state,
                                                        n_iter=hmm_n_iter, fillna=fillna)
        return hmm_disc_encoder
    
    elif method == "hmm_gen":
        hmm_gen_encoder = HMMGenerativeTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols,
                                                   num_cols=dynamic_num_cols, n_states=hmm_n_states,
                                                   min_seq_length=hmm_min_seq_length, max_seq_length=hmm_max_seq_length,
                                                   random_state=random_state, n_iter=hmm_n_iter, fillna=fillna)
        return hmm_gen_encoder
    
    else:
        print("Invalid encoder type")
        return None
    

##### MAIN PART ######   

outfile = os.path.join(home_dir, results_dir, "cv_results_%s_%s.csv"%(method_name, dataset_ref)) 

with open(outfile, 'w') as fout:

    fout.write("%s;%s;%s;%s;%s;%s;%s;%s\n"%("part", "dataset", "method", "rf_max_features", "hmm_n_states", "nr_events", "metric", "score"))
    
    for dataset_name in datasets:

        print("Reading dataset...")
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

        # maximum prefix length considered can't be larger than the 75th quantile
        min_prefix_length = 1 if "hmm_disc" not in methods else 2
        prefix_lengths = list(range(min_prefix_length, min(20, int(np.ceil(data.groupby(case_id_col).size().quantile(0.75)))) + 1))
        prefix_lengths = range(2,3)

        # split into train and test using temporal split
        start_timestamps = data.groupby(case_id_col)[timestamp_col].min().reset_index()
        start_timestamps.sort_values(timestamp_col, ascending=1, inplace=True)
        train_ids = list(start_timestamps[case_id_col])[:int(train_ratio*len(start_timestamps))]
        del start_timestamps
        train = data[data[case_id_col].isin(train_ids)].sort_values(timestamp_col, ascending=True)
        del data
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
            train_chunk = train[train[case_id_col].isin(current_train_names)]
            test_chunk = train[~train[case_id_col].isin(current_train_names)]

            grouped_train = train_chunk.sort_values(timestamp_col, ascending=True).groupby(case_id_col)
            grouped_test = test_chunk.sort_values(timestamp_col, ascending=True).groupby(case_id_col)

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


            for hmm_n_states in hmm_n_statess:
                print("Starting HMM with %s states..."%hmm_n_states)
                sys.stdout.flush()
                
                # test separately for each prefix length
                for nr_events in prefix_lengths:
                    print("Fitting pipeline for %s events..."%nr_events)
                    sys.stdout.flush()
                    
                    #### SELECT RELEVANT CASES ####
                    relevant_train_case_names = train_case_lengths[train_case_lengths == nr_events].index
                    relevant_test_case_names = test_case_lengths[test_case_lengths == nr_events].index

                    if len(relevant_test_case_names) == 0:
                        continue
                    elif len(relevant_train_case_names) == 0:
                        preds = [0.5] * len(relevant_test_case_names)
                        continue
                        
                    dt_train = train_prefixes[train_prefixes[case_id_col].isin(relevant_train_case_names)].sort_values(timestamp_col, ascending=True)
                    del relevant_train_case_names
                    train_y = dt_train.groupby(case_id_col).first()[label_col]
                    
                    dt_test = test_prefixes[test_prefixes[case_id_col].isin(relevant_test_case_names)].sort_values(timestamp_col, ascending=True)
                    del relevant_test_case_names
                    test_y = [1 if label==pos_label else 0 for label in dt_test.groupby(case_id_col).first()[label_col]]
                    
                    #### ENCODE DATA ####
                    start = time()
                    feature_combiner = FeatureUnion([(method, init_encoder(method)) for method in methods])
                    dt_train = feature_combiner.fit_transform(dt_train)
                    encode_time_train = time() - start
                    
                    start = time()
                    dt_test = feature_combiner.transform(dt_test)
                    encode_time_test = time() - start
                    
                    fout.write("%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, 0, hmm_n_states, nr_events, "encode_time_train", encode_time_train))
                    fout.write("%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, 0, hmm_n_states, nr_events, "encode_time_test", encode_time_test))
                    
                    for rf_max_features in rf_max_featuress:
                        print("Training RF with max features = %s..."%rf_max_features)
                        sys.stdout.flush()

                        #### FIT CLASSIFIER ####
                        cls = RandomForestClassifier(n_estimators=rf_n_estimators, max_features=rf_max_features, random_state=random_state)
                        start = time()
                        cls.fit(dt_train, train_y)
                        cls_fit_time = time() - start


                        #### PREDICT ####
                        start = time()
                        if len(cls.classes_) == 1:
                            hardcoded_prediction = 1 if cls.classes_[0] == pos_label else 0
                            preds = [hardcoded_prediction] * len(relevant_test_case_names)
                        else:
                            # make predictions
                            preds_pos_label_idx = np.where(cls.classes_ == pos_label)[0][0] 
                            preds = cls.predict_proba(dt_test)[:,preds_pos_label_idx]
                        prediction_time = time() - start

                        if len(set(test_y)) < 2:
                            auc = None
                        else:
                            auc = roc_auc_score(test_y, preds)
                        prec, rec, fscore, _ = precision_recall_fscore_support(test_y, [0 if pred < 0.5 else 1 for pred in preds], average="binary")

                        fout.write("%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_states, nr_events, "auc", auc))
                        fout.write("%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_states, nr_events, "precision", prec))
                        fout.write("%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_states, nr_events, "recall", rec))
                        fout.write("%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_states, nr_events, "fscore", fscore))
                        fout.write("%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_states, nr_events, "cls_fit_time", cls_fit_time))
                        fout.write("%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_states, nr_events, "cls_predict_time", prediction_time))

