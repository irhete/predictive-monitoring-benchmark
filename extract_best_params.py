import glob
import pandas as pd
import os
import numpy as np
import pickle
from sys import argv

cv_results_dir = argv[1] # cv_results
outfile = argv[2] #"optimal_params.pickle"
outfile_r = argv[3]

files = glob.glob("%s/*"%cv_results_dir)
files = [file for file in files if os.path.getsize(file) > 0]

data = pd.read_csv(files[0], sep=";")
for file in files[1:]:
    tmp = pd.read_csv(file, sep=";")
    data = pd.concat([data, tmp], axis=0)
        
data = data[data.metric=="auc"]
data["score"][pd.isnull(data["score"])] = 0

if data["score"].dtype != np.float64:
    data["score"][data["score"] == "None"] = 0

data["score"] = data["score"].astype(float)
data.fillna(0, inplace=True)

if "n_clusters" not in data.columns:
    data["n_clusters"] = 0

data_agg = data.groupby(["dataset", "hmm_n_states", "method", "metric", "n_clusters", "nr_events", "rf_max_features"], as_index=False)["score"].mean()
data_agg_over_all_prefixes = data.groupby(["dataset", "hmm_n_states", "method", "metric", "n_clusters", "rf_max_features"], as_index=False)["score"].mean()

data_best = data_agg.sort_values("score", ascending=False).groupby(["dataset", "method", "metric", "nr_events"], as_index=False).first()
data_best_over_all_prefixes = data_agg_over_all_prefixes.sort_values("score", ascending=False).groupby(["dataset", "method", "metric"], as_index=False).first()


best_params = {}

# index-based
for row in data_best[(data_best.method.str.contains("index")) & (~data_best.method.str.contains("single"))][["dataset", "method", "nr_events", "rf_max_features", "hmm_n_states"]].values:
    if row[0] not in best_params:
        best_params[row[0]] = {}
    if row[1] not in best_params[row[0]]:
        best_params[row[0]][row[1]] = {}
    if row[2] not in best_params[row[0]][row[1]]:
        best_params[row[0]][row[1]][row[2]] = {}
    best_params[row[0]][row[1]][row[2]]["rf_max_features"] = row[3] if row[3] == "sqrt" else float(row[3])
    best_params[row[0]][row[1]][row[2]]["hmm_n_states"] = int(row[4])
    
# single
for row in data_best_over_all_prefixes[data_best_over_all_prefixes.method.str.contains("single")][["dataset", "method", "rf_max_features", "hmm_n_states"]].values:
    if row[0] not in best_params:
        best_params[row[0]] = {}
    if row[1] not in best_params[row[0]]:
        best_params[row[0]][row[1]] = {}
    best_params[row[0]][row[1]]["rf_max_features"] = row[2] if row[2] == "sqrt" else float(row[2])
    best_params[row[0]][row[1]]["hmm_n_states"] = int(row[3])
    
# cluster
for row in data_best_over_all_prefixes[data_best_over_all_prefixes.method.str.contains("cluster")][["dataset", "method", "rf_max_features", "n_clusters", "hmm_n_states"]].values:
    if row[0] not in best_params:
        best_params[row[0]] = {}
    if row[1] not in best_params[row[0]]:
        best_params[row[0]][row[1]] = {}
    best_params[row[0]][row[1]]["rf_max_features"] = row[2] if row[2] == "sqrt" else float(row[2])
    best_params[row[0]][row[1]]["n_clusters"] = int(row[3])
    best_params[row[0]][row[1]]["hmm_n_states"] = int(row[4])
    
# hmm
for row in data_best_over_all_prefixes[data_best_over_all_prefixes.method.str.contains("hmm_static")][["dataset", "method", "rf_max_features", "hmm_n_states"]].values:
    if row[0] not in best_params:
        best_params[row[0]] = {}
    if row[1] not in best_params[row[0]]:
        best_params[row[0]][row[1]] = {}
    best_params[row[0]][row[1]]["rf_max_features"] = row[2] if row[2] == "sqrt" else float(row[2])
    best_params[row[0]][row[1]]["hmm_n_states"] = int(row[3])
 
tmp = data_best_over_all_prefixes[data_best_over_all_prefixes.method=="hmm_static"][["dataset", "method", "rf_max_features", "hmm_n_states"]]
tmp.to_csv(outfile_r, sep=";", index=False)

with open(outfile, "wb") as fout:
    pickle.dump(best_params, fout)