import pandas as pd
import numpy as np
import os
import sys

input_data_folder = "../orig_logs"
output_data_folder = "../labeled_logs_csv_processed"
in_filename = "Hospital Billing - Event Log.csv"

case_id_col = "Case ID"
activity_col = "Activity"
timestamp_col = "Complete Timestamp"
label_col = "label"
pos_label = "deviant"
neg_label = "regular"

category_freq_threshold = 10

# features for classifier
dynamic_cat_cols = ["Activity", 'Resource', 'actOrange', 'actRed', 'blocked', 'caseType', 'diagnosis', 'flagC', 'flagD', 'msgCode', 'msgType', 'state', 'version', 'isCancelled', 'isClosed', 'closeCode'] 
static_cat_cols = ['speciality']
dynamic_num_cols = ['msgCount']
static_num_cols = []

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

def check_if_attribute_exists(group, attribute, cut_from_idx=True):
    group[label_col] = neg_label if True in list(group[attribute]) else pos_label
    relevant_idxs = np.where(group[attribute]==True)[0]
    if len(relevant_idxs) > 0:
        cut_idx = relevant_idxs[0]
        if cut_from_idx:
            return group[:idx]
        else:
            return group
    else:
        return group
    
    
data = pd.read_csv(os.path.join(input_data_folder, in_filename), sep=";")
data[case_id_col] = data[case_id_col].fillna("missing_caseid")
data.rename(columns=lambda x: x.replace('(case) ', ''), inplace=True)

# remove incomplete cases
tmp = data.groupby(case_id_col).apply(check_if_any_of_activities_exist, activities=["BILLED", "DELETE", "FIN"])
incomplete_cases = tmp.index[tmp==False]
data = data[~data[case_id_col].isin(incomplete_cases)]
del tmp

data = data[static_cols + dynamic_cols]

# add features extracted from timestamp
data[timestamp_col] = pd.to_datetime(data[timestamp_col])
data["timesincemidnight"] = data[timestamp_col].dt.hour * 60 + data[timestamp_col].dt.minute
data["month"] = data[timestamp_col].dt.month
data["weekday"] = data[timestamp_col].dt.weekday
data["hour"] = data[timestamp_col].dt.hour
data = data.groupby(case_id_col).apply(extract_timestamp_features)

# add inter-case features
print("Extracting open cases...")
sys.stdout.flush()
data = data.sort_values([timestamp_col], ascending=True, kind='mergesort')
dt_first_last_timestamps = data.groupby(case_id_col)[timestamp_col].agg([min, max])
dt_first_last_timestamps.columns = ["start_time", "end_time"]
case_end_times = dt_first_last_timestamps.to_dict()["end_time"]

data["open_cases"] = 0
case_dict_state = {}
for idx, row in data.iterrows():
    case = row[case_id_col]
    current_ts = row[timestamp_col]

    # save the state
    data.set_value(idx, 'open_cases', len(case_dict_state))

    if current_ts >= case_end_times[case]:
        if case in case_dict_state:
            del case_dict_state[case]
    else:
        case_dict_state[case] = 1

print("Imputing missing values...")
sys.stdout.flush()
# impute missing values
grouped = data.sort_values(timestamp_col, ascending=True, kind='mergesort').groupby(case_id_col)
for col in static_cols + dynamic_cols:
    data[col] = grouped[col].transform(lambda grp: grp.fillna(method='ffill'))
        
data[cat_cols] = data[cat_cols].fillna('missing')
data = data.fillna(0)
    
# set infrequent factor levels to "other"
for col in cat_cols:
    counts = data[col].value_counts()
    mask = data[col].isin(counts[counts >= category_freq_threshold].index)
    data.loc[~mask, col] = "other"
    
data = data.sort_values(timestamp_col, ascending=True, kind="mergesort")    
    
# second labeling
dt_labeled = data.groupby(case_id_col).apply(check_if_attribute_exists, attribute="isClosed", cut_from_idx=False)
dt_labeled.drop(['isClosed'], axis=1).to_csv(os.path.join(output_data_folder, "hospital_billing_2.csv"), sep=";", index=False)
del dt_labeled
    
# third labeling
dt_labeled = data.groupby(case_id_col).apply(check_if_activity_exists, activity="REOPEN", cut_from_idx=True)
dt_labeled.to_csv(os.path.join(output_data_folder, "hospital_billing_3.csv"), sep=";", index=False)
del dt_labeled
    