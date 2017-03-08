import pandas as pd
import numpy as np
import sys
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import dataset_confs
import glob
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

datasets = ["bpic2011_f%s"%formula for formula in range(1,5)]
#datasets = ["bpic2015_%s_f%s"%(municipality, formula) for municipality in range(1,6) for formula in range(1,3)]
#datasets = ["insurance_activity", "insurance_followup"]
#datasets = ["traffic_fines_f%s"%formula for formula in range(1,4)]
#datasets = ["bpic2011_f%s"%formula for formula in range(1,5)] + ["bpic2015_%s_f%s"%(municipality, formula) for municipality in range(1,6) for formula in range(1,3)] + ["traffic_fines_f%s"%formula for formula in range(1,4)]
#datasets = ["bpic2011_f1"]
#datasets = ["sepsis_cases"]


prefix_lengths = list(range(2,21))

train_ratio = 0.8
max_len = 20
lstmsize = 64
dropout = 0
optim = 'rmsprop'
loss = 'binary_crossentropy'
nb_epoch = 30
batch_size = 1
time_dim = max_len
n_classes = 2

#outfile = "results/results_all_single_bpic2011.csv"
#outfile = "results/results_lstm_bpic2015.csv"
#outfile = "results/results_all_single_insurance.csv"
#outfile = "results/results_all_single_traffic.csv"
#outfile = "results/results_single_public.csv"
outfile = "results/results_lstm_bpic2011_lstmsize%s_dropout%s.csv"%(lstmsize, int(dropout*100))
#outfile = "results/results_lstm_sepsis_cases.csv"
    
    
##### MAIN PART ######    
with open(outfile, 'w') as fout:
    
    fout.write("%s;%s;%s;%s;%s\n"%("dataset", "method", "nr_events", "metric", "score"))
    
    for dataset_name in datasets:
        
        checkpoint_prefix = "checkpoints/%s_weights_lstmsize%s_dropout%s"%(dataset_name, lstmsize, int(dropout*100))
        checkpoint_filepath = "%s.{epoch:02d}-{val_loss:.2f}.hdf5"%checkpoint_prefix
        loss_file = "loss_files/%s_loss_lstmsize%s_dropout%s.txt"%(dataset_name, lstmsize, int(dropout*100))
        
        # read dataset settings
        case_id_col = dataset_confs.case_id_col[dataset_name]
        activity_col = dataset_confs.activity_col[dataset_name]
        timestamp_col = dataset_confs.timestamp_col[dataset_name]
        label_col = dataset_confs.label_col[dataset_name]
        pos_label = dataset_confs.pos_label[dataset_name]

        dynamic_cat_cols = dataset_confs.dynamic_cat_cols[dataset_name]
        static_cat_cols = dataset_confs.static_cat_cols[dataset_name]
        dynamic_num_cols = dataset_confs.dynamic_num_cols[dataset_name]
        static_num_cols = dataset_confs.static_num_cols[dataset_name]
        
        data_filepath = dataset_confs.filename[dataset_name]
        
        # specify data types
        dtypes = {col:"object" for col in dynamic_cat_cols+static_cat_cols+[case_id_col, label_col, timestamp_col]}
        for col in dynamic_num_cols + static_num_cols:
            dtypes[col] = "float"

        # read data
        data = pd.read_csv(data_filepath, sep=";", dtype=dtypes)
        data[timestamp_col] = pd.to_datetime(data[timestamp_col])

        # split into train and test using temporal split
        grouped = data.groupby(case_id_col)
        start_timestamps = grouped[timestamp_col].min().reset_index()
        start_timestamps.sort_values(timestamp_col, ascending=1, inplace=True)
        train_ids = list(start_timestamps[case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[case_id_col].isin(train_ids)].sort_values(timestamp_col, ascending=1)
        test = data[~data[case_id_col].isin(train_ids)].sort_values(timestamp_col, ascending=1)


        grouped_train = train.groupby(case_id_col)
        grouped_test = test.groupby(case_id_col)
        
        test_case_lengths = grouped_test.size()

        # encode data for LSTM
        print('Encoding training data...')
        # encode data for LSTM

        scaler = MinMaxScaler()

        dt_train_scaled = pd.DataFrame(scaler.fit_transform(train[dynamic_num_cols+static_num_cols]), index=train.index, columns=dynamic_num_cols+static_num_cols)
        dt_train_cat = pd.get_dummies(train[dynamic_cat_cols+static_cat_cols])
        dt_train = pd.concat([dt_train_scaled, dt_train_cat], axis=1)
        dt_train[case_id_col] = train[case_id_col]
        dt_train[label_col] = train[label_col]

        data_dim = dt_train.shape[1] - 2
        
        grouped = dt_train.groupby(case_id_col)
        X = np.zeros((0,max_len,data_dim))
        y = []
        for _, group in grouped:
            for i in range(1, min(max_len, len(group)) + 1):
                X = np.concatenate([X, pad_sequences(group.as_matrix()[np.newaxis,:i,:-2], maxlen=max_len)], axis=0)
            y.extend([group[label_col].iloc[0]] * (min(max_len, len(group))))

        
        y = pd.get_dummies(y)
        classes = y.columns
        y = y.as_matrix()
        
        method_name = "lstm"
            
        print('Build model...')
        model = Sequential()
        model.add(LSTM(lstmsize, input_shape=(time_dim, data_dim)))
        model.add(Dropout(dropout))
        model.add(Dense(n_classes, activation='softmax'))
        
        print('Compiling model...')
        model.compile(loss=loss, optimizer=optim)
        
        print("Training...")
        checkpointer = ModelCheckpoint(filepath=checkpoint_filepath, verbose=1, save_best_only=True, save_weights_only=True)
        history = model.fit(X, y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=2, validation_split=0.2, callbacks=[checkpointer])
        
        with open(loss_file, 'w') as fout2:
            fout2.write("epoch;train_loss;val_loss\n")
            for epoch in range(nb_epoch):
                fout2.write("%s;%s;%s\n"%(epoch, history.history['loss'][epoch], history.history['val_loss'][epoch]))
        
        
        # load the best weights
        lstm_weights_file = glob.glob("%s*.hdf5"%checkpoint_prefix)[-1]
        model.load_weights(lstm_weights_file)
        
        # test separately for each prefix length
        for nr_events in prefix_lengths:
            
            # select only cases that are at least of length nr_events
            relevant_case_ids = test_case_lengths.index[test_case_lengths >= nr_events]
            relevant_grouped_test = test[test[case_id_col].isin(relevant_case_ids)].sort_values(timestamp_col, ascending=True).groupby(case_id_col)
            test_y = pd.get_dummies(relevant_grouped_test.first()[label_col])[classes].as_matrix()
            dt_test = relevant_grouped_test.head(nr_events)

            dt_test_scaled = pd.DataFrame(scaler.fit_transform(dt_test[dynamic_num_cols+static_num_cols]), index=dt_test.index, columns=dynamic_num_cols+static_num_cols)
            dt_test_cat = pd.get_dummies(dt_test[dynamic_cat_cols+static_cat_cols])
            dt_test = pd.concat([dt_test[[case_id_col, label_col, timestamp_col]], dt_test_scaled, dt_test_cat], axis=1)

            # add missing columns if necessary
            missing_cols = [col for col in dt_train.columns if col not in dt_test.columns]
            for col in missing_cols:
                dt_test[col] = 0
            dt_test = dt_test[dt_train.columns]
            
            grouped = dt_test.groupby(case_id_col)

            test_X = np.zeros((0,max_len,data_dim))
            for _, group in grouped:
                test_X = np.concatenate([test_X, pad_sequences(group.as_matrix()[np.newaxis,:,:-2], maxlen=max_len)], axis=0)
                
            # predict    
            preds = model.predict(test_X)
            
            # evaluate
            if len(np.unique(test_y)) < 2:
                auc = None
            else:
                auc = roc_auc_score(test_y, preds)
                
            prec, rec, fscore, _ = precision_recall_fscore_support(test_y[:,np.where(classes==pos_label)[0][0]], [0 if pred < 0.5 else 1 for pred in preds[:,np.where(classes==pos_label)[0][0]]], average="binary")
                

            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "auc", auc))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "precision", prec))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "recall", rec))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, method_name, nr_events, "fscore", fscore))
                 
