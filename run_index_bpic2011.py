import pandas as pd
import numpy as np
import sys
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from transformers.StaticTransformer import StaticTransformer
from transformers.IndexBasedTransformer import IndexBasedTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion

datasets = {"bpic2011_f%s"%(formula):"labeled_logs_csv_processed/BPIC11_f%s.csv"%(formula) for formula in range(1,5)}
outfile = "results/results_index_bpic2011.csv"

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
static_num_cols = ["Age"]

train_ratio = 0.8
rf_n_estimators = 500
random_state = 22
fillna = True

methods_dict = {
    "index": ["static", "index"],
    "index_hmm_combined": ["static", "index", "hmm_disc"]}

def init_encoder(method):
    
    if method == "static":
        return StaticTransformer(case_id_col=case_id_col, cat_cols=static_cat_cols, num_cols=static_num_cols, fillna=fillna)
    
    elif method == "hmm_disc":
        hmm_disc_encoder = HMMDiscriminativeTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols,
                                                        num_cols=dynamic_num_cols, n_states=hmm_n_states, label_col=label_col,
                                                        pos_label=pos_label, min_seq_length=hmm_min_seq_length,
                                                        max_seq_length=hmm_max_seq_length, random_state=random_state,
                                                        n_iter=hmm_n_iter, fillna=fillna)
        hmm_disc_encoder.fit(train.sort_values(timestamp_col, ascending=True))
        return hmm_disc_encoder
    
    elif method == "index":
        return IndexBasedTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols,
                                     max_events=None, fillna=fillna)
    
    else:
        print("Invalid encoder type")
        return None
    
    
##### MAIN PART ######   
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
        
        train_y = grouped_train.first()[label_col]
        
        for method_name, methods in methods_dict.items():
            
            for nr_events in prefix_lengths:

                cls = RandomForestClassifier(n_estimators=rf_n_estimators, random_state=random_state)
                feature_combiner = FeatureUnion([(method, init_encoder(method)) for method in methods])
                pipeline = Pipeline([('encoder', feature_combiner), ('cls', cls)])

                # fit pipeline
                pipeline.fit(grouped_train.head(nr_events), train_y)
                preds_pos_label_idx = np.where(pipeline.named_steps["cls"].classes_ == pos_label)[0][0] 

                # test
                test_y = [1 if label==pos_label else 0 for label in grouped_test.first()[label_col]]
                preds = pipeline.predict_proba(grouped_test.head(nr_events))

                auc = roc_auc_score(test_y, preds[:,preds_pos_label_idx])
                prec, rec, fscore, _ = precision_recall_fscore_support(test_y, [0 if pred < 0.5 else 1 for pred in preds[:,preds_pos_label_idx]], average="binary")

                fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "auc", auc))
                fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "precision", prec))
                fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "recall", rec))
                fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "fscore", fscore))