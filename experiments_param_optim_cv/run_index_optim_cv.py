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
from transformers.IndexBasedTransformer import IndexBasedTransformer
from transformers.IndexBasedExtractor import IndexBasedExtractor
from transformers.HMMDiscriminativeTransformer import HMMDiscriminativeTransformer
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
    "index": ["static", "index"],
    "index_hmm_combined": ["static", "index", "hmm_disc"]}
    
methods = methods_dict[method_name]

prefix_lengths = list(range(1,21)) if "hmm_disc" not in methods else list(range(2,21))

train_ratio = 0.8
hmm_min_seq_length = 2
hmm_max_seq_length = None
hmm_n_iters = [0]#[10, 30, 50]
hmm_n_statess = [0]#[2, 5, 7, 10, 20, 40]
rf_n_estimators = 500
rf_max_featuress = ["sqrt", 0.05, 0.1, 0.25, 0.5, 0.75]
random_state = 22
fillna = True

##### MAIN PART ######   

outfile = os.path.join(home_dir, "cv_results/cv_results_index_%s.csv"%dataset_name)

with open(outfile, 'w') as fout:

    fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%("part", "dataset", "method", "rf_max_features", "hmm_n_iter", "hmm_n_states", "nr_events", "metric", "score"))

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
    
    part = 1
    for train_index, test_index in skf.split(grouped_train_firsts, grouped_train_firsts[label_col]):
        print("Starting chunk %s..."%part)
        sys.stdout.flush()
        
        # create train and validation data according to current fold
        current_train_names = grouped_train_firsts[case_id_col][train_index]
        train_chunk = train[train[case_id_col].isin(current_train_names)]
        test_chunk = train[~train[case_id_col].isin(current_train_names)]

        test_case_lengths = test_chunk.sort_values(timestamp_col, ascending=True).groupby(case_id_col).size()
        train_y = train_chunk.sort_values(timestamp_col, ascending=True).groupby(case_id_col).first()[label_col]
        test_y_all = test_chunk.sort_values(timestamp_col, ascending=True).groupby(case_id_col).first()[label_col]

        # encode all index-based
        print("Encoding index-based...")
        sys.stdout.flush()
        index_encoder = IndexBasedTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, max_events=prefix_lengths[-1], fillna=fillna)

        start = time()
        dt_train_index_all = index_encoder.transform(train_chunk)
        train_index_encode_time_all = time() - start

        start = time()
        dt_test_index_all = index_encoder.transform(test_chunk)
        test_index_encode_time_all = time() - start

        # encode all static
        print("Encoding static...")
        sys.stdout.flush()
        static_transformer = StaticTransformer(case_id_col=case_id_col, cat_cols=static_cat_cols, num_cols=static_num_cols, fillna=fillna)

        start = time()
        dt_train_static = static_transformer.transform(train_chunk)
        train_encode_time_base = time() - start

        start = time()
        dt_test_static = static_transformer.transform(test_chunk)
        test_encode_time_base = time() - start


        for rf_max_features in rf_max_featuress:
            for hmm_n_states in hmm_n_statess:
                for hmm_n_iter in hmm_n_iters:
                    print("RF max_features = %s, hmm_n_states = %s, hmm_n_iter = %s"%(rf_max_features, hmm_n_states, hmm_n_iter))
                    sys.stdout.flush()
                
                    for xx in range(1):
                        print("Evaluating method %s..."%method_name)
                        sys.stdout.flush()

                        fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_iter, hmm_n_states, 0, "train_index_encode_time_all", train_index_encode_time_all))
                        fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_iter, hmm_n_states, 0, "test_index_encode_time_all", test_index_encode_time_all))

                        for nr_events in prefix_lengths:

                            print("Evaluating for %s events..."%nr_events)
                            sys.stdout.flush()

                            # extract appropriate number of events for index-based encoding
                            index_extractor = IndexBasedExtractor(cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, max_events=nr_events, fillna=True)

                            start = time()
                            train_X = pd.concat([dt_train_static, index_extractor.transform(dt_train_index_all)], axis=1)
                            train_encode_time = train_encode_time_base + time() - start


                            # retain only test cases with length at least nr_events
                            relevant_test_static = dt_test_static[test_case_lengths >= nr_events]
                            if len(relevant_test_static) == 0:
                                break

                            start = time()
                            test_X = pd.concat([relevant_test_static, index_extractor.transform(dt_test_index_all.loc[test_case_lengths >= nr_events])], axis=1)
                            test_encode_time = test_encode_time_base + time() - start

                            test_y = [1 if label==pos_label else 0 for label in test_y_all[test_case_lengths >= nr_events]]

                    
                            # fit and encode HMM if needed
                            if "hmm_disc" in methods:
                                start = time()
                                hmm_disc_encoder = HMMDiscriminativeTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols,
                                                                    num_cols=dynamic_num_cols, n_states=hmm_n_states,
                                                                                label_col=label_col,
                                                                    pos_label=pos_label, min_seq_length=hmm_min_seq_length,
                                                                    max_seq_length=nr_events, random_state=random_state,
                                                                    n_iter=hmm_n_iter, fillna=fillna)
                                train_X = pd.concat([train_X, hmm_disc_encoder.fit_transform(train_chunk.groupby(case_id_col).head(nr_events))], axis=1)
                                train_encode_time += time() - start

                                relevant_test_ids = test_case_lengths.index[test_case_lengths >= nr_events]
                                relevant_grouped_test = test_chunk[test_chunk[case_id_col].isin(relevant_test_ids)].groupby(case_id_col, as_index=False)
                                start = time()
                                test_X = pd.concat([test_X, hmm_disc_encoder.transform(relevant_grouped_test.head(nr_events))], axis=1)
                                test_encode_time += time() - start
                                del relevant_grouped_test
                                del relevant_test_ids

                            fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_iter, hmm_n_states, nr_events, "nrow_train_X", train_X.shape[0]))
                            fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_iter, hmm_n_states, nr_events, "ncol_train_X", train_X.shape[1]))
                            fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_iter, hmm_n_states, nr_events, "nrow_test_X", test_X.shape[0]))
                            fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_iter, hmm_n_states, nr_events, "ncol_test_X", test_X.shape[1]))
                            
                            # fit classifier
                            print("Fitting classifier...")
                            sys.stdout.flush()
                            start = time()
                            cls = RandomForestClassifier(n_estimators=rf_n_estimators, max_features=rf_max_features, random_state=random_state)
                            cls.fit(train_X, train_y)
                            cls_fit_time = time() - start
                            preds_pos_label_idx = np.where(cls.classes_ == pos_label)[0][0] 
                            del train_X

                            # test
                            print("Testing...")
                            sys.stdout.flush()
                            start = time()
                            preds = cls.predict_proba(test_X)[:,preds_pos_label_idx]
                            cls_pred_time = time() - start
                            del test_X

                            if len(set(test_y)) < 2:
                                auc = None
                            else:
                                auc = roc_auc_score(test_y, preds)

                            prec, rec, fscore, _ = precision_recall_fscore_support(test_y, [0 if pred < 0.5 else 1 for pred in preds], average="binary")

                            fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_iter, hmm_n_states, nr_events, "auc", auc))
                            fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_iter, hmm_n_states, nr_events, "precision", prec))
                            fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_iter, hmm_n_states, nr_events, "recall", rec))
                            fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_iter, hmm_n_states, nr_events, "fscore", fscore))
                            fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_iter, hmm_n_states, nr_events, "train_encode_time", train_encode_time))
                            fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_iter, hmm_n_states, nr_events, "test_encode_time", test_encode_time))
                            fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_iter, hmm_n_states, nr_events, "cls_fit_time", cls_fit_time))
                            fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, rf_max_features, hmm_n_iter, hmm_n_states, nr_events, "cls_predict_time", cls_pred_time))
                            
        part += 1
