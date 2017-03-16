import pandas as pd
import numpy as np
import sys
from time import time
from sys import argv
import os

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

sys.path.append("..")
from transformers.StaticTransformer import StaticTransformer
from transformers.LastStateTransformer import LastStateTransformer
from transformers.PreviousStateTransformer import PreviousStateTransformer
from transformers.AggregateTransformer import AggregateTransformer
from transformers.IndexBasedTransformer import IndexBasedTransformer
from transformers.HMMDiscriminativeTransformer import HMMDiscriminativeTransformer
from transformers.HMMGenerativeTransformer import HMMGenerativeTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
import dataset_confs

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

dataset_name = argv[1]
method_name = argv[2]

home_dir = ".."

methods_dict = {
    "single_laststate": ["static", "laststate"],
    "single_last_two_states": ["static", "laststate", "prevstate"],
    "single_agg": ["static", "agg"],
    "single_hmm_disc": ["static", "hmm_disc"],
    "single_last_two_states_agg": ["static", "laststate", "prevstate", "agg"],
    "single_combined": ["static", "laststate", "agg", "hmm_disc"]}


methods = methods_dict[method_name]

prefix_lengths = list(range(1,21)) if "hmm_disc" not in methods else list(range(2,21))

train_ratio = 0.8
hmm_min_seq_length = 2
hmm_max_seq_length = None
hmm_n_iters = [0] if "hmm_disc" not in methods else [10, 30, 50]
hmm_n_statess = [0] if "hmm_disc" not in methods else list(range(1,11))
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
    
    
outfile = os.path.join(home_dir, "cv_results/cv_results_single_%s_%s.csv"%(dataset_name, method_name)) 
    
##### MAIN PART ######    
with open(outfile, 'w') as fout:
    
    fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%("part", "dataset", "method", "rf_max_features", "hmm_n_iter", "hmm_n_states", "nr_events", "metric", "score"))
    
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

    # specify data types
    dtypes = {col:"object" for col in dynamic_cat_cols+static_cat_cols+[case_id_col, label_col, timestamp_col]}
    for col in dynamic_num_cols + static_num_cols:
        dtypes[col] = "float"

    # read data
    print("Reading...")
    sys.stdout.flush()
    data = pd.read_csv(data_filepath, sep=";", dtype=dtypes)
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    print("Splitting...")
    sys.stdout.flush()
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

    part = 1
    for train_index, test_index in skf.split(grouped_train_firsts, grouped_train_firsts[label_col]):
        print("Starting chunk %s..."%part)
        sys.stdout.flush()

        # create train and validation data according to current fold
        current_train_names = grouped_train_firsts[case_id_col][train_index]
        train_chunk = train[train[case_id_col].isin(current_train_names)]
        test_chunk = train[~train[case_id_col].isin(current_train_names)]

        grouped_train = train_chunk.sort_values(timestamp_col, ascending=True).groupby(case_id_col)
        test_case_lengths = test_chunk.sort_values(timestamp_col, ascending=True).groupby(case_id_col).size()


        # generate prefix data (each possible prefix becomes a trace)
        print("Generating prefix data...")
        sys.stdout.flush()
        train_prefixes = grouped_train.head(prefix_lengths[0])
        for nr_events in prefix_lengths[1:]:
            tmp = grouped_train.head(nr_events)
            tmp[case_id_col] = tmp[case_id_col].apply(lambda x: "%s_%s"%(x, nr_events))
            train_prefixes = pd.concat([train_prefixes, tmp], axis=0)
        del grouped_train

        for rf_max_features in rf_max_featuress:
            for hmm_n_states in hmm_n_statess:
                for hmm_n_iter in hmm_n_iters:
                    print("RF max_features = %s, hmm_n_states = %s, hmm_n_iter = %s"%(rf_max_features, hmm_n_states, hmm_n_iter))
                    sys.stdout.flush()
                    

                    cls = RandomForestClassifier(n_estimators=rf_n_estimators, max_features=rf_max_features, random_state=random_state)
                    feature_combiner = FeatureUnion([(method, init_encoder(method)) for method in methods])
                    pipeline = Pipeline([('encoder', feature_combiner), ('cls', cls)])

                    # fit pipeline
                    train_y = train_prefixes.groupby(case_id_col).first()[label_col]

                    print("Fitting pipeline...")
                    sys.stdout.flush()
                    start = time()
                    pipeline.fit(train_prefixes, train_y)
                    pipeline_fit_time = time() - start

                    preds_pos_label_idx = np.where(pipeline.named_steps["cls"].classes_ == pos_label)[0][0] 

                    # get training times
                    train_encoding_fit_time = sum([el[1].fit_time for el in pipeline.named_steps["encoder"].transformer_list])
                    train_encoding_transform_time = sum([el[1].transform_time for el in pipeline.named_steps["encoder"].transformer_list])
                    cls_fit_time = pipeline_fit_time - train_encoding_fit_time - train_encoding_transform_time

                    fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_iter, hmm_n_states, 0, "cls_fit_time", cls_fit_time))
                    fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_iter, hmm_n_states, 0, "encoder_fit_time", train_encoding_fit_time))
                    fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_iter, hmm_n_states, 0, "encoder_transform_time_train", train_encoding_transform_time))
        
                    # test separately for each prefix length
                    for nr_events in prefix_lengths:

                        # select only cases that are at least of length nr_events
                        relevant_case_ids = test_case_lengths.index[test_case_lengths >= nr_events]
                        if len(relevant_case_ids) == 0:
                            break
                        relevant_grouped_test = test_chunk[test_chunk[case_id_col].isin(relevant_case_ids)].sort_values(timestamp_col, ascending=True).groupby(case_id_col, as_index=False)
                        del relevant_case_ids
                        test_y = [1 if label==pos_label else 0 for label in relevant_grouped_test.first()[label_col]]

                        # predict
                        print("Predicting for %s events..."%nr_events)
                        sys.stdout.flush()
                        start = time()
                        preds = pipeline.predict_proba(relevant_grouped_test.head(nr_events))[:,preds_pos_label_idx]
                        pipeline_pred_time = time() - start
                        del relevant_grouped_test
                        test_encoding_transform_time = sum([el[1].transform_time for el in pipeline.named_steps["encoder"].transformer_list])
                        cls_pred_time = pipeline_pred_time - test_encoding_transform_time

                        if len(set(test_y)) < 2:
                            auc = None
                        else:
                            auc = roc_auc_score(test_y, preds)
                        prec, rec, fscore, _ = precision_recall_fscore_support(test_y, [0 if pred < 0.5 else 1 for pred in preds], average="binary")

                        fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_iter, hmm_n_states, nr_events, "auc", auc))
                        fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_iter, hmm_n_states, nr_events, "precision", prec))
                        fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_iter, hmm_n_states, nr_events, "recall", rec))
                        fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_iter, hmm_n_states, nr_events, "fscore", fscore))
                        fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_iter, hmm_n_states, nr_events, "cls_predict_time", cls_pred_time))
                        fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_iter, hmm_n_states, nr_events, "encoder_transform_time_test", test_encoding_transform_time))

                    print("\n")
                
                
