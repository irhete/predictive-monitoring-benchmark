import pandas as pd
import numpy as np
import os

input_data_folder = "../orig_logs"
output_data_folder = "../labeled_logs_csv_processed"
filenames = ["Production.csv"]

case_id_col = "Case ID"
activity_col = "Activity"
resource_col = "Resource"
timestamp_col = "Complete Timestamp"
label_col = "label"
pos_label = "deviant"
neg_label = "regular"

freq_threshold = 10
timeunit = 'm'

# features for classifier
static_cat_cols = ["Part_Desc_", "Rework"]
static_num_cols = ["Work_Order_Qty"]
dynamic_cat_cols = [activity_col, resource_col, "Report_Type", "Resource.1"]
dynamic_num_cols = ["Qty_Completed", "Qty_for_MRB", "activity_duration"]

static_cols = static_cat_cols + static_num_cols + [case_id_col, label_col]
dynamic_cols = dynamic_cat_cols + dynamic_num_cols + [timestamp_col]
cat_cols = dynamic_cat_cols + static_cat_cols

def assign_label(group):
    tmp = group["Qty_Rejected"] > 0
    tmp = tmp.reset_index()["Qty_Rejected"]
    if sum(tmp) > 0:
        idx = tmp[tmp==True].index[0]
        group = group.iloc[:idx,:]
        group[label_col] = pos_label
    else:
        group[label_col] = neg_label
    return group

def extract_timestamp_features(group):
    
    group = group.sort_values(timestamp_col, ascending=False, kind='mergesort')
    
    tmp = group[timestamp_col] - group[timestamp_col].shift(-1)
    tmp = tmp.fillna(0)
    group["timesincelastevent"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 'm'))) # m is for minutes

    tmp = group[timestamp_col] - group[timestamp_col].iloc[-1]
    tmp = tmp.fillna(0)
    group["timesincecasestart"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 'm'))) # m is for minutes

    group = group.sort_values(timestamp_col, ascending=True, kind='mergesort')
    group["event_nr"] = range(1, len(group) + 1)
    
    return group

def get_open_cases(date):
    return sum((dt_first_last_timestamps["start_time"] <= date) & (dt_first_last_timestamps["end_time"] > date))


for filename in filenames:
    
    data = pd.read_csv(os.path.join(input_data_folder,filename), sep=";")

    # add event duration
    data["Complete Timestamp"] = pd.to_datetime(data["Complete Timestamp"])
    data["Start Timestamp"] = pd.to_datetime(data["Start Timestamp"])
    tmp = data["Complete Timestamp"] - data["Start Timestamp"]
    tmp = tmp.fillna(0)
    data["activity_duration"] = tmp.apply(lambda x: float(x / np.timedelta64(1, timeunit)))
    
    # assign labels
    data = data.sort_values(timestamp_col, ascending=True, kind='mergesort').groupby(case_id_col).apply(assign_label)
    
    data = data[static_cols + dynamic_cols]

    # add features extracted from timestamp
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    data["timesincemidnight"] = data[timestamp_col].dt.hour * 60 + data[timestamp_col].dt.minute
    data["month"] = data[timestamp_col].dt.month
    data["weekday"] = data[timestamp_col].dt.weekday
    data["hour"] = data[timestamp_col].dt.hour
    data = data.groupby(case_id_col).apply(extract_timestamp_features)
    
    # add inter-case features
    data = data.sort_values([timestamp_col], ascending=True, kind='mergesort')
    dt_first_last_timestamps = data.groupby(case_id_col)[timestamp_col].agg([min, max])
    dt_first_last_timestamps.columns = ["start_time", "end_time"]
    data["open_cases"] = data[timestamp_col].apply(get_open_cases)
    
    # impute missing values
    grouped = data.sort_values(timestamp_col, ascending=True, kind='mergesort').groupby(case_id_col)
    for col in static_cols + dynamic_cols:
        data[col] = grouped[col].transform(lambda grp: grp.fillna(method='ffill'))
        
    data[cat_cols] = data[cat_cols].fillna('missing')
    data = data.fillna(0)
        
    # set infrequent factor levels to "other"
    for col in cat_cols:
        counts = data[col].value_counts()
        mask = data[col].isin(counts[counts >= freq_threshold].index)
        data.loc[~mask, col] = "other"

    data.to_csv(os.path.join(output_data_folder,filename), sep=";", index=False)
    