from sklearn.ensemble import RandomForestClassifier
from IndexBasedTransformer import IndexBasedTransformer
from sklearn.metrics import roc_auc_score
import pandas as pd
from HMMTransformer import HMMTransformer

datasets = {"bpic2011_f1":"labeled_logs_csv/BPIC11_f1.csv", "bpic2011_f2":"labeled_logs_csv/BPIC11_f2.csv", "bpic2011_f3":"labeled_logs_csv/BPIC11_f3.csv", "bpic2011_f4":"labeled_logs_csv/BPIC11_f4.csv"}
outfile = "results_hmm_bpic2011.csv"

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

train_ratio = 0.8
n_states = 3
n_iter = 20

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


        for nr_events in prefix_lengths:
            hmm_transformer = HMMTransformer(n_states, dynamic_cols, cat_cols, case_id_col, timestamp_col, label_col, pos_label, min_seq_length=2, max_seq_length=nr_events, random_state=22, n_iter=n_iter)
            hmm_transformer.fit(train)
            
            index_encoder = IndexBasedTransformer(nr_events, static_cols, dynamic_cols, cat_cols, case_id_col, timestamp_col, label_col, activity_col)
            tmp = index_encoder.transform(train)
            
            dt_hmm = hmm_transformer.transform(train)
            hmm_merged = tmp.merge(dt_hmm, on=case_id_col)
            X = hmm_merged.drop([label_col, case_id_col], axis=1)
            y = hmm_merged[label_col]
            
            cls = RandomForestClassifier(n_estimators=500, random_state=22)
            cls.fit(X, y)


            test_encoded = index_encoder.transform(test)
            test_encoded = test_encoded.merge(hmm_transformer.transform(test), on=case_id_col)
            preds = cls.predict_proba(test_encoded.drop([label_col, case_id_col], axis=1))

            score = roc_auc_score([1 if label==pos_label else 0 for label in test_encoded[label_col]], preds[:,0])
            
            fout.write("%s;%s;%s\n"%(dataset_name, nr_events, score))