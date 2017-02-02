import pandas as pd
import numpy as np
import os

input_data_folder = "labeled_logs_csv"
output_data_folder = "labeled_logs_csv_processed"
filenames = ["traffic_fines_f%s.csv"%i for i in range(1,4)]


case_id_col = "Case ID"
activity_col = "Activity"
timestamp_col = "Complete Timestamp"
label_col = "label"
pos_label = "deviant"
neg_label = "regular"

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


for filename in filenames:
    data = pd.read_csv(os.path.join(input_data_folder,filename), sep=";")
    
    data.rename(columns=lambda x: x.replace('(case) ', ''), inplace=True)
    
    # replace infrequent dismissals with "other"
    data.loc[data["dismissal"] != "NIL", "dismissal"] = "other"
    
    data = data[static_cols + dynamic_cols]
    
    # add features extracted from timestamp
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    data = data.groupby(case_id_col).apply(extract_timestamp_features)
    
    # impute missing values
    grouped = data.sort_values(timestamp_col, ascending=True).groupby(case_id_col)
    for col in static_cols + dynamic_cols:
        data[col] = grouped[col].transform(lambda grp: grp.fillna(method='ffill'))
        
    data[cat_cols] = data[cat_cols].fillna('missing')
    data = data.fillna(0)
    
    data.to_csv(os.path.join(output_data_folder,filename), sep=";", index=False)
    