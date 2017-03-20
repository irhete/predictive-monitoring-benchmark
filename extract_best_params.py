import glob
import pandas as pd
import os
import numpy as np
import pickle
from sys import argv

cv_results_dir = argv[1] # cv_results
outfile = argv[2] #"optimal_params.pickle"

files = glob.glob("%s/*"%cv_results_dir)

data = pd.read_csv(files[0], sep=";")
for file in files[1:]:
    if os.path.getsize(file) > 0:
        tmp = pd.read_csv(file, sep=";")
        data = pd.concat([data, tmp], axis=0)
        
data = data[data.metric=="auc"]
#data["score"][pd.isnull(data["score"])] = 0
#data["score"][data["score"] == "None"] = 0
#data["score"] = data["score"].astype(float)
data.fillna(0, inplace=True)


data_agg = data.groupby(["dataset", "hmm_n_iter", "hmm_n_states", "method", "metric", "n_clusters", "nr_events", "rf_max_features"], as_index=False)["score"].mean()
data_agg_over_all_prefixes = data.groupby(["dataset", "hmm_n_iter", "hmm_n_states", "method", "metric", "n_clusters", "rf_max_features"], as_index=False)["score"].mean()

data_best = data_agg.sort_values("score", ascending=False).groupby(["dataset", "method", "metric", "nr_events"], as_index=False).first()
data_best_over_all_prefixes = data_agg_over_all_prefixes.sort_values("score", ascending=False).groupby(["dataset", "method", "metric"], as_index=False).first()


best_params = {}

# index-based
for row in data_best[data_best.method=="index"][["dataset", "method", "nr_events", "rf_max_features"]].values:
    if row[0] not in best_params:
        best_params[row[0]] = {}
    if row[1] not in best_params[row[0]]:
        best_params[row[0]][row[1]] = {}
    if row[2] not in best_params[row[0]][row[1]]:
        best_params[row[0]][row[1]][row[2]] = {}
    best_params[row[0]][row[1]][row[2]]["rf_max_features"] = row[3] if row[3] == "sqrt" else float(row[3])
    
# single
for row in data_best_over_all_prefixes[data_best_over_all_prefixes.method.isin(["single_laststate", "single_agg"])][["dataset", "method", "rf_max_features"]].values:
    if row[0] not in best_params:
        best_params[row[0]] = {}
    if row[1] not in best_params[row[0]]:
        best_params[row[0]][row[1]] = {}
    best_params[row[0]][row[1]]["rf_max_features"] = row[2] if row[2] == "sqrt" else float(row[2])
    
# cluster
for row in data_best_over_all_prefixes[data_best_over_all_prefixes.method.isin(["cluster_laststate", "cluster_agg"])][["dataset", "method", "rf_max_features", "n_clusters"]].values:
    if row[0] not in best_params:
        best_params[row[0]] = {}
    if row[1] not in best_params[row[0]]:
        best_params[row[0]][row[1]] = {}
    best_params[row[0]][row[1]]["rf_max_features"] = row[2] if row[2] == "sqrt" else float(row[2])
    best_params[row[0]][row[1]]["n_clusters"] = int(row[3])
    

with open(outfile, "wb") as fout:
    pickle.dump(best_params, fout)