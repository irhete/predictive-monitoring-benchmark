import pandas as pd
import numpy as np
import sys
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from time import time
import pickle
import os
from sys import argv

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

home_dir = ".."

with open(os.path.join(home_dir, "optimal_params.pickle"), "rb") as fin:
    best_params = pickle.load(fin)


dataset_ref_to_datasets = {
    "bpic2011": ["bpic2011_f%s"%formula for formula in range(1,5)],
    "bpic2015_f1": ["bpic2015_%s_f1"%(municipality) for municipality in range(1,6)],
    "bpic2015_f2": ["bpic2015_%s_f2"%(municipality) for municipality in range(1,6)],
    "insurance": ["insurance_activity", "insurance_followup"],
    "sepsis_cases": ["sepsis_cases"],
    "traffic_fines": ["traffic_fines"],
    "siae": ["siae"],
    "bpic2017": ["bpic2017"]
}
    

dataset_ref = argv[1]
datasets = dataset_ref_to_datasets[dataset_ref]
outfile = os.path.join(home_dir, "final_results/final_results_index_%s.csv"%dataset_ref)

prefix_lengths = list(range(1,21))

train_ratio = 0.8
hmm_min_seq_length = 2
hmm_max_seq_length = None
hmm_n_iter = 30
hmm_n_states = 6
rf_n_estimators = 500
random_state = 22
fillna = True

methods_dict = {
    "index": ["static", "index"]}#,
    #"index_hmm_combined": ["static", "index", "hmm_disc"]}
   
    
##### MAIN PART ######   
with open(outfile, 'w') as fout:
    
    fout.write("%s;%s;%s;%s;%s\n"%("dataset", "method", "nr_events", "metric", "score"))
    
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

        # split into train and test using temporal split
        start_timestamps = data.groupby(case_id_col)[timestamp_col].min().reset_index()
        start_timestamps.sort_values(timestamp_col, ascending=1, inplace=True)
        train_ids = list(start_timestamps[case_id_col])[:int(train_ratio*len(start_timestamps))]
        del start_timestamps
        train = data[data[case_id_col].isin(train_ids)].sort_values(timestamp_col, ascending=True)
        test = data[~data[case_id_col].isin(train_ids)].sort_values(timestamp_col, ascending=True)
        del data
        del train_ids

        test_case_lengths = test.sort_values(timestamp_col, ascending=True).groupby(case_id_col).size()
        train_y = train.sort_values(timestamp_col, ascending=True).groupby(case_id_col).first()[label_col]
        test_y_all = test.sort_values(timestamp_col, ascending=True).groupby(case_id_col).first()[label_col]
        
        # encode all index-based
        print("Encoding index-based...")
        sys.stdout.flush()
        index_encoder = IndexBasedTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols,
                                     max_events=prefix_lengths[-1], fillna=fillna)
        
        start = time()
        dt_train_index_all = index_encoder.transform(train)
        train_index_encode_time_all = time() - start
        
        start = time()
        dt_test_index_all = index_encoder.transform(test)
        test_index_encode_time_all = time() - start
        
        # encode all static
        print("Encoding static...")
        sys.stdout.flush()
        static_transformer = StaticTransformer(case_id_col=case_id_col, cat_cols=static_cat_cols, num_cols=static_num_cols, fillna=fillna)
        
        start = time()
        dt_train_static = static_transformer.transform(train)
        train_encode_time_base = time() - start
        
        start = time()
        dt_test_static = static_transformer.transform(test)
        test_encode_time_base = time() - start
        
        for method_name, methods in methods_dict.items():
            print("Evaluating method %s..."%method_name)
            sys.stdout.flush()
            
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, 0, "train_index_encode_time_all", train_index_encode_time_all))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, 0, "test_index_encode_time_all", test_index_encode_time_all))
            
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
                                                        num_cols=dynamic_num_cols, n_states=hmm_n_states, label_col=label_col,
                                                        pos_label=pos_label, min_seq_length=hmm_min_seq_length,
                                                        max_seq_length=nr_events, random_state=random_state,
                                                        n_iter=hmm_n_iter, fillna=fillna)
                    train_X = pd.concat([train_X, hmm_disc_encoder.fit_transform(train.groupby(case_id_col).head(nr_events))], axis=1)
                    train_encode_time += time() - start
                    
                    relevant_test_ids = test_case_lengths.index[test_case_lengths >= nr_events]
                    relevant_grouped_test = test[test[case_id_col].isin(relevant_test_ids)].groupby(case_id_col, as_index=False)
                    start = time()
                    test_X = pd.concat([test_X, hmm_disc_encoder.transform(relevant_grouped_test.head(nr_events))], axis=1)
                    test_encode_time += time() - start
                    del relevant_grouped_test
                    del relevant_test_ids
                
                # fit classifier
                print("Fitting classifier...")
                sys.stdout.flush()
                start = time()
                cls = RandomForestClassifier(n_estimators=rf_n_estimators, max_features=best_params[dataset_name][method_name][nr_events]['rf_max_features'], random_state=random_state)
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

                fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "auc", auc))
                fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "precision", prec))
                fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "recall", rec))
                fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "fscore", fscore))
                fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "train_encode_time", train_encode_time))
                fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "test_encode_time", test_encode_time))
                fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "cls_fit_time", cls_fit_time))
                fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "cls_predict_time", cls_pred_time))
