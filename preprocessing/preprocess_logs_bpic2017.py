import pandas as pd
import numpy as np
import os

input_data_folder = "../labeled_logs_csv"
output_data_folder = "../labeled_logs_csv_processed"
filenames = ["BPIC17.csv"]

case_id_col = "Case ID"
activity_col = "Activity"
timestamp_col = "Complete Timestamp"
label_col = "label"
pos_label = "deviant"
neg_label = "regular"

#category_freq_threshold = 100
max_category_levels = 10

# features for classifier
dynamic_cat_cols = ["Activity", 'Resource', 'Action', 'CreditScore', 'EventOrigin', 'lifecycle:transition'] # i.e. event attributes
static_cat_cols = ['ApplicationType', 'LoanGoal'] # i.e. case attributes that are known from the start
dynamic_num_cols = ['FirstWithdrawalAmount', 'MonthlyCost', 'NumberOfTerms', 'OfferedAmount', "activity_duration"]
static_num_cols = ['RequestedAmount']

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


def assign_label(group):
    group[label_col] = neg_label if True in list(group["Accepted"]) else pos_label
    return group


def cut_before_activity(group):
    relevant_activity_idxs = np.where(group[activity_col] == relevant_activity)[0]
    if len(relevant_activity_idxs) > 0:
        cut_idx = relevant_activity_idxs[0]
        return group[:cut_idx]
    else:
        return group


for filename in filenames:
    
    data = pd.read_csv(os.path.join(input_data_folder,filename), sep=";")
    data.rename(columns=lambda x: x.replace('(case) ', ''), inplace=True)

    # add event duration
    data["Complete Timestamp"] = pd.to_datetime(data["Complete Timestamp"])
    data["Start Timestamp"] = pd.to_datetime(data["Start Timestamp"])
    tmp = data["Complete Timestamp"] - data["Start Timestamp"]
    tmp = tmp.fillna(0)
    data["activity_duration"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 'm')))
    
    # assign class labels
    data = data.groupby(case_id_col).apply(assign_label)

    data = data[static_cols + dynamic_cols]

    # add features extracted from timestamp
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    data = data.groupby(case_id_col).apply(extract_timestamp_features)
    
    #print("Cutting activities...")
    # cut traces before relevant activity happens
    #relevant_activity = "O_Accepted"
    #data = data.sort_values(timestamp_col).groupby(case_id_col).apply(cut_before_activity)
    
    # impute missing values
    grouped = data.sort_values(timestamp_col, ascending=True).groupby(case_id_col)
    for col in static_cols + dynamic_cols:
        data[col] = grouped[col].transform(lambda grp: grp.fillna(method='ffill'))
        
    data[cat_cols] = data[cat_cols].fillna('missing')
    data = data.fillna(0)
        
    # set infrequent factor levels to "other"
    for col in cat_cols:
        if col != activity_col:
            counts = data[col].value_counts()
            #mask = data[col].isin(counts[counts >= category_freq_threshold].index)
            #data.loc[~mask, col] = "other"
            mask = data[col].isin(counts.index[max_category_levels:])
            data.loc[mask, col] = "other"

    data.to_csv(os.path.join(output_data_folder,filename), sep=";", index=False)
    