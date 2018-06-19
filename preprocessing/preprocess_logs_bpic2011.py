import pandas as pd
import numpy as np
import os

input_data_folder = "../labeled_logs_csv"
output_data_folder = "../labeled_logs_csv_processed"
filenames = ["BPIC11_f%s.csv"%(formula) for formula in range(1,5)]

case_id_col = "Case ID"
activity_col = "Activity code"
resource_col = "Producer code"
timestamp_col = "time:timestamp"
label_col = "label"
pos_label = "deviant"
neg_label = "regular"

category_freq_threshold = 10

# features for classifier
dynamic_cat_cols = ["Activity code", "Producer code", "Section", "Specialism code.1", "group"]
static_cat_cols = ["Diagnosis", "Treatment code", "Diagnosis code", "Specialism code", "Diagnosis Treatment Combination ID"]
dynamic_num_cols = ["Number of executions"]
static_num_cols = ["Age"]

static_cols = static_cat_cols + static_num_cols + [case_id_col, label_col]
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


def cut_before_activity(group):
    relevant_activity_idxs = np.where(group[activity_col] == relevant_activity)[0]
    if len(relevant_activity_idxs) > 0:
        cut_idx = relevant_activity_idxs[0]
        return group[:cut_idx]
    else:
        return group

def get_open_cases(date):
    return sum((dt_first_last_timestamps["start_time"] <= date) & (dt_first_last_timestamps["end_time"] > date))

for filename in filenames:
    data = pd.read_csv(os.path.join(input_data_folder,filename), sep=";")

    # switch labels (deviant/regular was set incorrectly before)
    data = data.set_value(col=label_col, index=(data[label_col] == pos_label), value="normal")
    data = data.set_value(col=label_col, index=(data[label_col] == neg_label), value=pos_label)
    data = data.set_value(col=label_col, index=(data[label_col] == "normal"), value=neg_label)

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
    
    print("Cutting activities...")
    # cut traces before relevant activity happens
    if "f1" in filename:
        relevant_activity = "AC379414" #"tumor marker CA-19.9"
        data = data.sort_values(timestamp_col, kind='mergesort').groupby(case_id_col).apply(cut_before_activity)
        relevant_activity = "378619A" #"ca-125 using meia"
        data = data.sort_values(timestamp_col, kind='mergesort').groupby(case_id_col).apply(cut_before_activity)
        
    elif "f3" in filename:
        relevant_activity = "AC356134" #"histological examination - biopsies nno"
        data = data.sort_values(timestamp_col, kind='mergesort').groupby(case_id_col).apply(cut_before_activity)
        relevant_activity = "376480A" #"squamous cell carcinoma using eia"
        data = data.sort_values(timestamp_col, kind='mergesort').groupby(case_id_col).apply(cut_before_activity)
        
    elif "f4" in filename:
        relevant_activity = "AC356133" #"histological examination - big resectiep"
        data = data.sort_values(timestamp_col, kind='mergesort').groupby(case_id_col).apply(cut_before_activity)
    
    
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
    
    data.to_csv(os.path.join(output_data_folder, filename), sep=";", index=False)
    