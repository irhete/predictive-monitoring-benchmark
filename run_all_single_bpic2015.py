import pandas as pd
import numpy as np
import sys
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from transformers.StaticTransformer import StaticTransformer
from transformers.LastStateTransformer import LastStateTransformer
from transformers.AggregateTransformer import AggregateTransformer
from transformers.IndexBasedTransformer import IndexBasedTransformer
from transformers.HMMDiscriminativeTransformer import HMMDiscriminativeTransformer
from transformers.HMMGenerativeTransformer import HMMGenerativeTransformer
from sklearn.ensemble import RandomForestClassifier

datasets = {"bpic2015_%s_f%s"%(municipality, formula):"labeled_logs_csv_processed/BPIC15_%s_f%s.csv"%(municipality, formula) for municipality in range(1,6) for formula in range(1,3)}
outfile = "results_all_single_bpic2015.csv"

prefix_lengths = list(range(2,21))

case_id_col = "Case ID"
activity_col = "Activity"
timestamp_col = "Complete Timestamp"
label_col = "label"
pos_label = "deviant"
neg_label = "regular"

# features for classifier
dynamic_cat_cols_base = ["Activity", "monitoringResource", "question", "Resource"]
static_cat_cols = ["Responsible_actor"]
dynamic_num_cols = ["duration", "month", "weekday", "hour"]
static_num_cols = ["SUMleges", label_col]

train_ratio = 0.8
min_seq_length = 2
max_seq_length = None
random_state = 22
n_iter = 30
n_states = 6

methods = ["laststate", "agg", "hmm_disc", "hmm_gen", "combined"]
clss = {method:RandomForestClassifier(n_estimators=500, random_state=22) for method in methods}

with open(outfile, 'w') as fout:
    
    fout.write("%s;%s;%s;%s;%s\n"%("dataset", "method", "nr_events", "metric", "score"))
    
    for dataset_name, data_filepath in datasets.items():
        data = pd.read_csv(data_filepath, sep=";")
        
        # add "parts" columns
        dynamic_cat_cols = dynamic_cat_cols_base + [col for col in data.columns if col not in dynamic_cat_cols_base+static_cat_cols+dynamic_num_cols+static_num_cols+[timestamp_col, case_id_col]]

        # split into train and test using temporal split
        grouped = data.groupby(case_id_col)
        start_timestamps = grouped[timestamp_col].min().reset_index()
        start_timestamps.sort_values(timestamp_col, ascending=1, inplace=True)
        train_ids = list(start_timestamps[case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[case_id_col].isin(train_ids)]
        test = data[~data[case_id_col].isin(train_ids)]

        grouped_train = train.sort_values(timestamp_col, ascending=True).groupby(case_id_col)
        grouped_test = test.sort_values(timestamp_col, ascending=True).groupby(case_id_col)

        train_prefixes = grouped_train.head(prefix_lengths[0])
        for nr_events in prefix_lengths[1:]:
            tmp = grouped_train.head(nr_events)
            tmp[case_id_col] = tmp[case_id_col].apply(lambda x: "%s_%s"%(x, nr_events))
            train_prefixes = pd.concat([train_prefixes, tmp], axis=0)
        
        static_transformer = StaticTransformer(case_id_col=case_id_col, timestamp_col=timestamp_col, cat_cols=static_cat_cols, num_cols=static_num_cols, fillna=True)
        dt_static = static_transformer.fit_transform(train_prefixes)
        dt_test_static = static_transformer.transform(test)
        
        for method in methods:
            ### Last state ###
            if method == "laststate":
                last_state_transformer = LastStateTransformer(case_id_col=case_id_col, timestamp_col=timestamp_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, fillna=True)
                dt_last_state = last_state_transformer.fit_transform(train_prefixes)
                dt_train = dt_static.merge(dt_last_state, on=case_id_col)

            ### Aggregated ###
            elif method == "agg":
                aggregate_transformer = AggregateTransformer(case_id_col=case_id_col, timestamp_col=timestamp_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, fillna=True)
                dt_aggregate = aggregate_transformer.fit_transform(train_prefixes)
                dt_train = dt_static.merge(dt_aggregate, on=case_id_col)

            ### HMM discriminative ###
            elif method == "hmm_disc":
                hmm_discriminative_transformer = HMMDiscriminativeTransformer(case_id_col=case_id_col, timestamp_col=timestamp_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, n_states=n_states, label_col=label_col, pos_label=pos_label, min_seq_length=min_seq_length, max_seq_length=max_seq_length, random_state=random_state, n_iter=n_iter, fillna=True)
                hmm_discriminative_transformer.fit(train)
                dt_hmm_disc = hmm_discriminative_transformer.transform(train_prefixes)
                dt_train = dt_static.merge(dt_hmm_disc, on=case_id_col)

            ### HMM generative ###
            elif method == "hmm_gen":
                hmm_generative_transformer = HMMGenerativeTransformer(case_id_col=case_id_col, timestamp_col=timestamp_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, n_states=n_states, min_seq_length=min_seq_length, max_seq_length=max_seq_length, random_state=random_state, n_iter=n_iter, fillna=True)
                hmm_generative_transformer.fit(train)
                dt_hmm_gen = hmm_generative_transformer.transform(train_prefixes)
                dt_train = dt_static.merge(dt_hmm_gen, on=case_id_col)
            
            ### Combined ###
            elif method == "combined":
                dt_train = dt_static.merge(dt_last_state, on=case_id_col)
                dt_train = dt_train.merge(dt_aggregate, on=case_id_col)
                dt_train = dt_train.merge(dt_hmm_disc, on=case_id_col)
                dt_train = dt_train.merge(dt_hmm_gen, on=case_id_col)
        
            # fit classifier
            clss[method].fit(dt_train.drop([case_id_col, label_col], axis=1), dt_train[label_col])

        
        # test
        for nr_events in prefix_lengths:
            dt_test_prefix = grouped_test.head(nr_events)
            
            for method in methods:
                ### Last state ###
                if method == "laststate":
                    dt_test_last_state = last_state_transformer.transform(dt_test_prefix)
                    dt_test = dt_test_static.merge(dt_test_last_state, on=case_id_col)

                ### Aggregated ###
                elif method == "agg":
                    dt_test_aggregate = aggregate_transformer.transform(dt_test_prefix)
                    dt_test = dt_test_static.merge(dt_test_aggregate, on=case_id_col)

                ### HMM discriminative ###
                elif method == "hmm_disc":
                    dt_test_hmm_disc = hmm_discriminative_transformer.transform(dt_test_prefix)
                    dt_test = dt_test_static.merge(dt_test_hmm_disc, on=case_id_col)

                ### HMM generative ###
                elif method == "hmm_gen":
                    dt_test_hmm_gen = hmm_generative_transformer.transform(dt_test_prefix)
                    dt_test = dt_test_static.merge(dt_test_hmm_gen, on=case_id_col)

                ### Combined ###
                elif method == "combined":
                    dt_test = dt_test_static.merge(dt_test_last_state, on=case_id_col)
                    dt_test = dt_test.merge(dt_test_aggregate, on=case_id_col)
                    dt_test = dt_test.merge(dt_test_hmm_disc, on=case_id_col)
                    dt_test = dt_test.merge(dt_test_hmm_gen, on=case_id_col)
            
            
                preds_pos_label_idx = np.where(clss[method].classes_ == pos_label)[0][0]  
                preds = clss[method].predict_proba(dt_test.drop([case_id_col, label_col], axis=1))

                auc = roc_auc_score([1 if label==pos_label else 0 for label in dt_test[label_col]], preds[:,preds_pos_label_idx])
                prec, rec, fscore, _ = precision_recall_fscore_support([1 if label==pos_label else 0 for label in dt_test[label_col]], [0 if pred < 0.5 else 1 for pred in preds[:,preds_pos_label_idx]], average="binary")

                fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method, nr_events, "auc", auc))
                fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method, nr_events, "precision", prec))
                fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method, nr_events, "recall", rec))
                fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method, nr_events, "fscore", fscore))