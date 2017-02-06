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

datasets = {"bpic2011_f%s"%(formula):"labeled_logs_csv_processed/BPIC11_f%s.csv"%(formula) for formula in range(1,5)}
outfile = "results/results_combined_single_bpic2011.csv"

prefix_lengths = list(range(2,21))

case_id_col = "Case ID"
activity_col = "Activity code"
timestamp_col = "Complete Timestamp"
label_col = "label"
pos_label = "deviant"
neg_label = "regular"

# features for classifier
dynamic_cat_cols = ["Activity code", "Producer code", "Section", "Specialism code", "group"]
static_cat_cols = ["Diagnosis", "Treatment code", "Diagnosis code", "case Specialism code", "Diagnosis Treatment Combination ID"]
dynamic_num_cols = ["Number of executions", "duration", "month", "weekday", "hour"]
static_num_cols = ["Age", label_col]

train_ratio = 0.8
min_seq_length = 2
max_seq_length = None
random_state = 22
n_iter = 30
n_states = 6
n_clusters = 20
clustering_method = "kmeans"

methods = ["combined"]
#methods = ["laststate", "agg", "hmm_disc", "hmm_gen", "combined"]

with open(outfile, 'w') as fout:
    
    fout.write("%s;%s;%s;%s;%s\n"%("dataset", "method", "nr_events", "metric", "score"))
    
    for dataset_name, data_filepath in datasets.items():
        data = pd.read_csv(data_filepath, sep=";")

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
        
        
        # fit model (cluster traces and build classifier for each cluster)
        data_encoder = LastStateTransformer(case_id_col=case_id_col, timestamp_col=timestamp_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, fillna=True)
        model = ClusteringPredictiveModel(case_id_col, activity_col, label_col, timestamp_col, cat_cols, numeric_cols, n_clusters=n_clusters, n_estimators=500, random_state=22, fillna=True, pos_label=pos_label, clustering_method=clustering_method, data_encoder=data_encoder)
        model.fit(dt_prefixes)
        
        static_transformer = StaticTransformer(case_id_col=case_id_col, timestamp_col=timestamp_col, cat_cols=static_cat_cols, num_cols=static_num_cols, fillna=True)
        dt_static = static_transformer.fit_transform(train_prefixes)
        dt_test_static = static_transformer.transform(test)
        
        # test
        for nr_events in prefix_lengths:
            dt_test_prefix = grouped_test.head(nr_events)
            
            if method != "combined":
                dt_test = dt_test_static.merge(dynamic_transformer.transform(dt_test_prefix), on=case_id_col)
            else:
                dt_test = dt_test_static.merge(dynamic_transformer1.transform(dt_test_prefix), on=case_id_col)
                dt_test = dt_test.merge(dynamic_transformer2.transform(dt_test_prefix), on=case_id_col)
                        

            preds_pos_label_idx = np.where(cls.classes_ == pos_label)[0][0]  
            preds = cls.predict_proba(dt_test.drop([case_id_col, label_col], axis=1))

            auc = roc_auc_score([1 if label==pos_label else 0 for label in dt_test[label_col]], preds[:,preds_pos_label_idx])
            prec, rec, fscore, _ = precision_recall_fscore_support([1 if label==pos_label else 0 for label in dt_test[label_col]], [0 if pred < 0.5 else 1 for pred in preds[:,preds_pos_label_idx]], average="binary")
            
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method, nr_events, "auc", auc))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method, nr_events, "precision", prec))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method, nr_events, "recall", rec))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method, nr_events, "fscore", fscore))