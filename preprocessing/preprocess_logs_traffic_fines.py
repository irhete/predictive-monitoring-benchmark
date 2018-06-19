import pandas as pd
import numpy as np
import os
import sys

input_data_folder = "../orig_logs"
output_data_folder = "../labeled_logs_csv_processed"
filenames = ["Road_Traffic_Fine_Management_Process.csv"]

case_id_col = "Case ID"
activity_col = "Activity"
resource_col = "Resource"
timestamp_col = "Complete Timestamp"
label_col = "label"
pos_label = "deviant"
neg_label = "regular"

freq_threshold = 10

# features for classifier
dynamic_cat_cols = ["Activity", "Resource", "lastSent", "notificationType", "dismissal"]
static_cat_cols = ["article", "vehicleClass"]
dynamic_num_cols = ["expense"]
static_num_cols = ["amount", "points"]

static_cols = static_cat_cols + static_num_cols + [case_id_col]
dynamic_cols = dynamic_cat_cols + dynamic_num_cols + [timestamp_col]
cat_cols = dynamic_cat_cols + static_cat_cols


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

def check_if_activity_exists(group, activity, cut_from_idx=True):
    relevant_activity_idxs = np.where(group[activity_col] == activity)[0]
    if len(relevant_activity_idxs) > 0:
        idx = relevant_activity_idxs[0]
        group[label_col] = pos_label
        if cut_from_idx:
            return group[:idx]
        else:
            return group
    else:
        group[label_col] = neg_label
        return group   

for filename in filenames:
    print("Starting...")
    sys.stdout.flush()
    data = pd.read_csv(os.path.join(input_data_folder,filename), sep=";")
    
    data.rename(columns=lambda x: x.replace('(case) ', ''), inplace=True)
    
    # discard incomplete cases
    last_events = data.sort_values([timestamp_col], ascending=True, kind='mergesort').groupby(case_id_col).last()["Activity"]
    incomplete_cases = last_events.index[last_events=="Send Fine"]
    data = data[~data[case_id_col].isin(incomplete_cases)]
    
    # add event duration
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    data["timesincemidnight"] = data[timestamp_col].dt.hour * 60 + data[timestamp_col].dt.minute
    data["month"] = data[timestamp_col].dt.month
    data["weekday"] = data[timestamp_col].dt.weekday
    data["hour"] = data[timestamp_col].dt.hour
    
    # add features extracted from timestamp
    print("Extracting timestamp features...")
    sys.stdout.flush()
    data = data.groupby(case_id_col).apply(extract_timestamp_features)
    
    # add inter-case features
    print("Extracting open cases...")
    sys.stdout.flush()
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
        if col != activity_col:
            counts = data[col].value_counts()
            mask = data[col].isin(counts[counts >= freq_threshold].index)
            data.loc[~mask, col] = "other"
    
    data = data[static_cols + dynamic_cols]
    
    # assign class labels
    print("Assigning class labels...")
    sys.stdout.flush()
    data = data.sort_values([timestamp_col], ascending=True, kind='mergesort')
    dt_labeled = data.groupby(case_id_col).apply(check_if_activity_exists, activity="Send for Credit Collection", cut_from_idx=True)
    dt_labeled.to_csv(os.path.join(output_data_folder, "traffic_fines_1.csv"), sep=";", index=False)
    
    