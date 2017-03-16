import pandas as pd
import numpy as np
import os
import sys

input_data_folder = "../orig_logs"
output_data_folder = "../labeled_logs_csv_processed"
filenames = ["Road_Traffic_Fine_Management_Process.csv"]


case_id_col = "Case ID"
activity_col = "Activity"
timestamp_col = "Complete Timestamp"
label_col = "label"
pos_label = "deviant"
neg_label = "regular"

#category_freq_threshold = 10
max_category_levels = 10


# features for classifier
dynamic_cat_cols = ["Activity", "Resource", "lastSent", "notificationType", "dismissal"]
static_cat_cols = ["article", "vehicleClass"]
dynamic_num_cols = ["expense"]
static_num_cols = ["amount", "points"]

static_cols = static_cat_cols + static_num_cols + [case_id_col]
dynamic_cols = dynamic_cat_cols + dynamic_num_cols + [timestamp_col]
cat_cols = dynamic_cat_cols + static_cat_cols


def extract_timestamp_features(group):
    
    group = group.sort_values(timestamp_col, ascending=False)
    start_date = group[timestamp_col].iloc[-1]
    
    tmp = group[timestamp_col] - group[timestamp_col].shift(-1)
    tmp = tmp.fillna(0)
    group["duration"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 'm'))) # m is for minutes
    
    group["month"] = group[timestamp_col].dt.month
    group["weekday"] = group[timestamp_col].dt.weekday
    group["hour"] = group[timestamp_col].dt.hour
    
    return group


def assign_label(group):
    relevant_activity_idxs = np.where(group[activity_col] == "Payment")[0]
    if len(relevant_activity_idxs) > 0:
        cut_idx = relevant_activity_idxs[0]
        group[label_col] = pos_label
        return group[:cut_idx]
    else:
        group[label_col] = neg_label
        return group


for filename in filenames:
    print("Starting...")
    sys.stdout.flush()
    data = pd.read_csv(os.path.join(input_data_folder,filename), sep=";")
    
    data.rename(columns=lambda x: x.replace('(case) ', ''), inplace=True)
    
    data = data[static_cols + dynamic_cols]
        
    print("Adding timestamp features...")
    sys.stdout.flush()
    # add features extracted from timestamp
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    data = data.groupby(case_id_col).apply(extract_timestamp_features)
    
    print("Assigning labels...")
    sys.stdout.flush()
    # cut traces before relevant activity happens
    data = data.sort_values(timestamp_col, ascending=True).groupby(case_id_col).apply(assign_label)
    
    print("imputing missing values...")
    sys.stdout.flush()
    # impute missing values
    grouped = data.sort_values(timestamp_col, ascending=True).groupby(case_id_col)
    for col in static_cols + dynamic_cols:
        data[col] = grouped[col].transform(lambda grp: grp.fillna(method='ffill'))
        
    data[cat_cols] = data[cat_cols].fillna('missing')
    data = data.fillna(0)
    
    print("Renaming infrequent levels...")
    sys.stdout.flush()
    # set infrequent factor levels to "other"
    for col in cat_cols:
        counts = data[col].value_counts()
        #mask = data[col].isin(counts[counts >= category_freq_threshold].index)
        #data.loc[~mask, col] = "other"
        mask = data[col].isin(counts.index[max_category_levels:])
        data.loc[mask, col] = "other"
    
    data.to_csv(os.path.join(output_data_folder,filename), sep=";", index=False)
    