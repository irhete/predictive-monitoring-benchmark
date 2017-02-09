import pandas as pd
import numpy as np
import os

input_data_folder = "../labeled_logs_csv"
output_data_folder = "../labeled_logs_csv_processed"
filenames = ["traffic_fines_f%s.csv"%i for i in range(1,4)]


case_id_col = "Case ID"
activity_col = "Activity"
timestamp_col = "Complete Timestamp"
label_col = "label"
pos_label = "deviant"
neg_label = "regular"

category_freq_threshold = 10


# features for classifier
dynamic_cat_cols = ["Activity", "Resource", "lastSent", "notificationType", "dismissal"]
static_cat_cols = ["article", "vehicleClass"]
dynamic_num_cols = ["expense"]
static_num_cols = ["amount", "points"]

static_cols = static_cat_cols + static_num_cols + [case_id_col, label_col]
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


def cut_before_activity(group):
    relevant_activity_idxs = np.where(group[activity_col] == relevant_activity)[0]
    if len(relevant_activity_idxs) > 0:
        cut_idx = relevant_activity_idxs[0]
        return group[:cut_idx]
    else:
        return group


for filename in filenames:
    print("Starting...")
    data = pd.read_csv(os.path.join(input_data_folder,filename), sep=";")
    
    data.rename(columns=lambda x: x.replace('(case) ', ''), inplace=True)
    
    data = data[static_cols + dynamic_cols]
    
    print("Adding timestamp features...")
    # add features extracted from timestamp
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    data = data.groupby(case_id_col).apply(extract_timestamp_features)
    
    print(data.shape)
    print("Cutting activities...")
    # cut traces before relevant activity happens
    if "f1" in filename:
        relevant_activity = "Payment"
        data = data.sort_values(timestamp_col).groupby(case_id_col).apply(cut_before_activity)
    elif "f3" in filename:
        relevant_activity = "Add penalty"
        data = data.sort_values(timestamp_col).groupby(case_id_col).apply(cut_before_activity)
    print(data.shape)
    
    
    
    print("imputing missing values...")
    # impute missing values
    grouped = data.sort_values(timestamp_col, ascending=True).groupby(case_id_col)
    for col in static_cols + dynamic_cols:
        data[col] = grouped[col].transform(lambda grp: grp.fillna(method='ffill'))
        
    data[cat_cols] = data[cat_cols].fillna('missing')
    data = data.fillna(0)
    
    print("Renaming infrequent levels...")
    # set infrequent factor levels to "other"
    for col in cat_cols:
        counts = data[col].value_counts()
        mask = data[col].isin(counts[counts >= category_freq_threshold].index)
        data.loc[~mask, col] = "other"
    
    data.to_csv(os.path.join(output_data_folder,filename), sep=";", index=False)
    