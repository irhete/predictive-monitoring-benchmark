import pandas as pd
import time
import numpy as np
import sys
sys.path.append("../clustering-based-predictive-monitoring/")
from ClusteringPredictiveModel import ClusteringPredictiveModel
from sklearn.metrics import roc_auc_score

datasets = {"bpic2011_f1":"labeled_logs_csv/BPIC11_f1.csv", "bpic2011_f2":"labeled_logs_csv/BPIC11_f2.csv", "bpic2011_f3":"labeled_logs_csv/BPIC11_f3.csv", "bpic2011_f4":"labeled_logs_csv/BPIC11_f4.csv"}
outfile = "results_clustering_bpic2011.csv"

prefix_lengths = list(range(2,21))

case_id_col = "Case ID"
activity_col = "Activity code"
timestamp_col = "Complete Timestamp"
label_col = "label"
pos_label = "deviant"
neg_label = "regular"
dynamic_cols = ["Number of executions", "Producer code", "Section", "Specialism code", "group"] # i.e. event attributes
static_cols = ["Age", "Diagnosis", "Treatment code", "Diagnosis code", "case Specialism code", "Diagnosis Treatment Combination ID"] # i.e. case attributes that are known from the start
cat_cols = ["Activity code", "Producer code", "Section", "group", "Diagnosis", "Specialism code", "Treatment code", "Diagnosis code", "case Specialism code", "Diagnosis Treatment Combination ID"]
numeric_cols = ['Number of executions', "Age"]

train_ratio = 0.8
n_clusters = 20

with open(outfile, 'w') as fout:
    for dataset_name, data_filepath in datasets.items():
        data = pd.read_csv(data_filepath, sep=";")

        data.rename(columns = {'(case) Specialism code':'case Specialism code'}, inplace=True)
        data.rename(columns=lambda x: x.replace('(case) ', ''), inplace=True)

        # switch labels (deviant/regular was set incorrectly before)
        data = data.set_value(col=label_col, index=(data[label_col] == pos_label), value="normal")
        data = data.set_value(col=label_col, index=(data[label_col] == neg_label), value=pos_label)
        data = data.set_value(col=label_col, index=(data[label_col] == "normal"), value=neg_label)

        data = data[static_cols + dynamic_cols + [case_id_col, label_col, activity_col, timestamp_col]]

        data[cat_cols] = data[cat_cols].fillna('missing')
        data = data.fillna(0)

        # split into train and test using temporal split
        grouped = data.groupby(case_id_col)
        start_timestamps = grouped[timestamp_col].min().reset_index()
        start_timestamps.sort_values(timestamp_col, ascending=1, inplace=True)
        train_ids = list(start_timestamps[case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[case_id_col].isin(train_ids)]
        test = data[~data[case_id_col].isin(train_ids)]

        grouped_train = train.sort_values(timestamp_col, ascending=True).groupby(case_id_col)
        grouped_test = test.sort_values(timestamp_col, ascending=True).groupby(case_id_col)

        dt_prefixes = grouped_train.head(1)
        for nr_events in prefix_lengths:
            tmp = grouped_train.head(nr_events)
            tmp[case_id_col] = tmp[case_id_col].apply(lambda x: "%s_%s"%(x, nr_events))
            dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)
        
        # fit model (cluster traces and build classifier for each cluster)
        model = ClusteringPredictiveModel(case_id_col, activity_col, label_col, timestamp_col, cat_cols, numeric_cols, n_clusters=n_clusters, n_estimators=500, random_state=22, fillna=True, pos_label=pos_label)
        model.fit(dt_prefixes)
            
            
        for nr_events in prefix_lengths:
            # predict for test set (assign each trace to the closest cluster and predict outcome using the corresponding classifier)
            preds = model.predict_proba(grouped_test.head(nr_events))

            score = roc_auc_score(model.actual, preds)
            
            fout.write("%s;%s;%s\n"%(dataset_name, nr_events, score))