import pandas as pd
import numpy as np
import sys
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from transformers.StaticTransformer import StaticTransformer
from transformers.IndexBasedTransformer import IndexBasedTransformer
from sklearn.ensemble import RandomForestClassifier

datasets = {"bpic2015_%s_f%s"%(municipality, formula):"labeled_logs_csv_processed/BPIC15_%s_f%s.csv"%(municipality, formula) for municipality in range(1,6) for formula in range(1,3)}
outfile = "results/results_index_bpic2015.csv"

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
random_state = 22
n_estimators = 500

method = "index"

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
        
        static_transformer = StaticTransformer(case_id_col=case_id_col, timestamp_col=timestamp_col, cat_cols=static_cat_cols, num_cols=static_num_cols, fillna=True)
        dt_static = static_transformer.fit_transform(train)
        dt_test_static = static_transformer.transform(test)
        
        for nr_events in prefix_lengths:
            dt_train_prefix = grouped_train.head(nr_events)
            dt_test_prefix = grouped_test.head(nr_events)

            index_based_transformer = IndexBasedTransformer(case_id_col=case_id_col, timestamp_col=timestamp_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, nr_events=nr_events, fillna=True)
            dt_index_based = index_based_transformer.fit_transform(data)
            dt_train = dt_static.merge(dt_index_based, on=case_id_col)
            
            # fit classifier
            cls = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
            cls.fit(dt_train.drop([case_id_col, label_col], axis=1), dt_train[label_col])
            
            dt_test_index = index_based_transformer.transform(dt_test_prefix)
            dt_test = dt_test_static.merge(dt_test_index, on=case_id_col)

            preds_pos_label_idx = np.where(cls.classes_ == pos_label)[0][0]  
            preds = cls.predict_proba(dt_test.drop([case_id_col, label_col], axis=1))

            auc = roc_auc_score([1 if label==pos_label else 0 for label in dt_test[label_col]], preds[:,preds_pos_label_idx])
            prec, rec, fscore, _ = precision_recall_fscore_support([1 if label==pos_label else 0 for label in dt_test[label_col]], [0 if pred < 0.5 else 1 for pred in preds[:,preds_pos_label_idx]], average="binary")

            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method, nr_events, "auc", auc))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method, nr_events, "precision", prec))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method, nr_events, "recall", rec))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method, nr_events, "fscore", fscore))
            sys.stdout.flush()