import pandas as pd
import numpy as np
import sys
from sklearn.metrics import roc_auc_score
from FrequencyEncoder import FrequencyEncoder
from LastStateEncoder import LastStateEncoder
from sklearn.ensemble import RandomForestClassifier


datasets = {"bpic2011_f1":"labeled_logs_csv/BPIC11_f1.csv", "bpic2011_f2":"labeled_logs_csv/BPIC11_f2.csv", "bpic2011_f3":"labeled_logs_csv/BPIC11_f3.csv", "bpic2011_f4":"labeled_logs_csv/BPIC11_f4.csv"}
outfile = "results_frequency_controlflow_bpic2011.csv"

prefix_lengths = list(range(2,21))

case_id_col = "Case ID"
activity_col = "Activity code"
timestamp_col = "Complete Timestamp"
label_col = "label"
pos_label = "deviant"
neg_label = "regular"
cat_cols = ["Activity code"]
numeric_cols = []

train_ratio = 0.8

with open(outfile, 'w') as fout:
    for dataset_name, data_filepath in datasets.items():
        data = pd.read_csv(data_filepath, sep=";")

        data.rename(columns = {'(case) Specialism code':'case Specialism code'}, inplace=True)
        data.rename(columns=lambda x: x.replace('(case) ', ''), inplace=True)

        # switch labels (deviant/regular was set incorrectly before)
        data = data.set_value(col=label_col, index=(data[label_col] == pos_label), value="normal")
        data = data.set_value(col=label_col, index=(data[label_col] == neg_label), value=pos_label)
        data = data.set_value(col=label_col, index=(data[label_col] == "normal"), value=neg_label)

        data = data[numeric_cols + cat_cols + [case_id_col, label_col, timestamp_col]]

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

        train_prefixes = grouped_train.head(prefix_lengths[0])
        for nr_events in prefix_lengths[1:]:
            tmp = grouped_train.head(nr_events)
            tmp[case_id_col] = tmp[case_id_col].apply(lambda x: "%s_%s"%(x, nr_events))
            train_prefixes = pd.concat([train_prefixes, tmp], axis=0)
        
        # encode data
        freq_encoder = FrequencyEncoder(case_id_col, cat_cols)
        last_state_encoder = LastStateEncoder(case_id_col, timestamp_col, cat_cols = [], numeric_cols = numeric_cols + [label_col], fillna = True)

        train_freqs = freq_encoder.fit_transform(train_prefixes)
        train_last_state = last_state_encoder.fit_transform(train_prefixes)
        dt_train = train_freqs.merge(train_last_state, on=case_id_col)
        
        # fit classifier
        cls = RandomForestClassifier(n_estimators=500, random_state=22)
        cls.fit(dt_train.drop([case_id_col, label_col], axis=1), dt_train[label_col])
        
        preds_pos_label_idx = np.where(cls.classes_ == pos_label)[0][0]    
            
        # test
        for nr_events in prefix_lengths:
            test_freqs = freq_encoder.transform(grouped_test.head(nr_events))
            test_last_state = last_state_encoder.transform(grouped_test.head(nr_events))
            dt_test = test_freqs.merge(test_last_state, on=case_id_col)
            preds = cls.predict_proba(dt_test.drop([case_id_col, label_col], axis=1))
            
            score = roc_auc_score([1 if label==pos_label else 0 for label in dt_test[label_col]], preds[:,preds_pos_label_idx])
            
            fout.write("%s;%s;%s\n"%(dataset_name, nr_events, score))