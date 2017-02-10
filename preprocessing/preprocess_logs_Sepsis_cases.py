import pandas as pd
import numpy as np
import os

input_data_folder = "../labeled_logs_csv"
output_data_folder = "../labeled_logs_csv_processed"
filenames = ["Sepsis Cases_label.csv"]

case_id_col = "Case ID"
activity_col = "Activity"
timestamp_col = "Complete Timestamp"
label_col = "label"
pos_label = "deviant"
neg_label = "regular"

category_freq_threshold = 10

# features for classifier
dynamic_cat_cols = ["Activity", 'Diagnose', 'org:group'] # i.e. event attributes
static_cat_cols = ['DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG',
       'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor',
       'DiagnosticOther', 'DiagnosticSputum', 'DiagnosticUrinaryCulture',
       'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg',
       'Hypotensie', 'Hypoxie', 'InfectionSuspected', 'Infusion', 'Oligurie',
       'SIRSCritHeartRate', 'SIRSCritLeucos', 'SIRSCritTachypnea',
       'SIRSCritTemperature', 'SIRSCriteria2OrMore'] # i.e. case attributes that are known from the start
dynamic_num_cols = ['CRP', 'LacticAcid', 'Leucocytes']
static_num_cols = ['Age']

label_col = "label"
pos_label = "deviant"
neg_label = "regular"

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

    # assign class labels
    grouped = data.groupby(case_id_col)
    tmp = grouped.apply(lambda x: "Bad" in x[activity_col].values)
    pos_cases = tmp.index[tmp==False]
    data[label_col] = neg_label
    data = data.set_value(data[case_id_col].isin(pos_cases), label_col, pos_label)
    data = data[~data[activity_col].isin(["Good", "Bad"])]

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
        
    # set infrequent factor levels to "other"
    for col in cat_cols:
        counts = data[col].value_counts()
        mask = data[col].isin(counts[counts >= category_freq_threshold].index)
        data.loc[~mask, col] = "other"
        
    data.to_csv(os.path.join(output_data_folder,filename), sep=";", index=False)
    