import pandas as pd
import numpy as np
import os
import sys

input_data_folder = "../orig_logs"
output_data_folder = "../labeled_logs_csv_processed"
filenames = ["bpic2012.csv"]

case_id_col = "Case ID"
activity_col = "Activity"
resource_col = "Resource"
timestamp_col = "Complete Timestamp"
label_col = "label"
pos_label = "deviant"
neg_label = "regular"

relevant_offer_events = ["O_CANCELLED-COMPLETE", "O_ACCEPTED-COMPLETE", "O_DECLINED-COMPLETE"]

freq_threshold = 10

# features for classifier
static_cat_cols = []
static_num_cols = ["AMOUNT_REQ"]
dynamic_cat_cols = [activity_col, resource_col, 'lifecycle:transition']
dynamic_num_cols = ["timesincemidnight", "timesincelastevent", "timesincecasestart", "event_nr", "month", "weekday", "hour", "open_cases"]

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

def get_open_cases(date):
    return sum((dt_first_last_timestamps["start_time"] <= date) & (dt_first_last_timestamps["end_time"] > date))


for filename in filenames:
    
    data = pd.read_csv(os.path.join(input_data_folder, filename), sep=";")
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    data[resource_col] = data.sort_values(timestamp_col, ascending=True).groupby(case_id_col)[resource_col].transform(lambda grp: grp.fillna(method='ffill'))
    data.rename(columns=lambda x: x.replace('(case) ', ''), inplace=True)

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
    
    # assign class labels
    last_o_events = data[data[activity_col].str.startswith("O")].sort_values(timestamp_col, ascending=True).groupby(case_id_col).last()[activity_col]
    last_o_events = pd.DataFrame(last_o_events)
    last_o_events.columns = ["last_o_activity"]
    data = data.merge(last_o_events, left_on=case_id_col, right_index=True)
    data = data[data.last_o_activity.isin(relevant_offer_events)]
    
    for activity in relevant_offer_events:
        dt_labeled = data.copy()
        dt_labeled[label_col] = neg_label
        dt_labeled.ix[dt_labeled["last_o_activity"] == activity, label_col] = pos_label
        
        dt_labeled = dt_labeled[static_cols + dynamic_cols]
        
        # impute missing values
        grouped = dt_labeled.sort_values(timestamp_col, ascending=True).groupby(case_id_col)
        for col in static_cols + dynamic_cols:
            dt_labeled[col] = grouped[col].transform(lambda grp: grp.fillna(method='ffill'))

        dt_labeled[cat_cols] = dt_labeled[cat_cols].fillna('missing')
        dt_labeled = dt_labeled.fillna(0)

        # set infrequent factor levels to "other"
        for col in cat_cols:
            counts = dt_labeled[col].value_counts()
            mask = dt_labeled[col].isin(counts[counts >= freq_threshold].index)
            dt_labeled.loc[~mask, col] = "other"

        dt_labeled.to_csv(os.path.join(output_data_folder, "%s_%s.csv" % (filename[:-4], activity)), sep=";", index=False)
    