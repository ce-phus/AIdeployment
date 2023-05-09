from django.shortcuts import render
from django.http import HttpRequest
from django.core.exceptions import ValidationError
import numpy as np
import os
import keras
from keras.models import load_model
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Activation
import pandas as pd
from werkzeug.utils import secure_filename
from sklearn import preprocessing
from keras.layers import Dense, Dropout, LSTM
# Create your views here.

UPLOAD_FOLDER = 'uploads'
ALLOWED_FILE_TYPES = 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif' 

def r2_keras(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

model = keras.models.load_model('C:/Users/Nakitare/Documents/project/AI/models/WS100_dropout_repeat.h5',custom_objects={'r2_keras': r2_keras})


def model_predict(file_path, model):
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['Altitude', 'Mach_No', 'TRA']
    sensor_names = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30','phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
    col_names = index_names + setting_names + sensor_names
    train = pd.read_csv(('train_FD004.txt'), sep='\s+', header=None, names=col_names)
    test = pd.read_csv((file_path), sep='\s+', header=None, names=col_names)
    y_test = pd.read_csv(('RUL_FD004.txt'), sep='\s+', header=None, names=['RemainingUsefulLife'])
    train = train.sort_values(['unit_nr', 'time_cycles'])
    rul = pd.DataFrame(train.groupby('unit_nr')['time_cycles'].max()).reset_index()
    rul.columns = ['unit_nr', 'max']
    train = train.merge(rul, on=['unit_nr'], how='left')
    train['RUL'] = train['max'] - train['time_cycles']
    train.drop('max', axis=1, inplace=True)
    w1 = 30
    w0 = 15
    train['label1'] = np.where(train['RUL'] <= w1, 1, 0)
    train['label2'] = train['label1']
    train.loc[train['RUL'] <= w0, 'label2'] = 2
    train['cycle_norm'] = train['time_cycles']
    cols_normalize = train.columns.difference(
    ['unit_nr', 'time_cycles', 'RUL', 'label1', 'label2'])
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train[cols_normalize]), columns=cols_normalize, index=train.index)
    join_df = train[train.columns.difference(cols_normalize)].join(norm_train_df)
    train = join_df.reindex(columns=train.columns)
    test['cycle_norm'] = test['time_cycles']
    norm_test_df = pd.DataFrame(min_max_scaler.transform(test[cols_normalize]), columns=cols_normalize, index=test.index)
    test_join_df = test[test.columns.difference(cols_normalize)].join(norm_test_df)
    test = test_join_df.reindex(columns=test.columns)
    test = test.reset_index(drop=True)
    rul = pd.DataFrame(test.groupby('unit_nr')['time_cycles'].max()).reset_index()
    rul.columns = ['unit_nr', 'max']
    y_test.columns = ['more']
    y_test['unit_nr'] = y_test.index + 1
    y_test['max'] = rul['max'] + y_test['more']
    y_test.drop('more', axis=1, inplace=True)
    test = test.merge(y_test, on=['unit_nr'], how='left')
    test['RUL'] = test['max'] - test['time_cycles']
    test.drop('max', axis=1, inplace=True)
    test['label1'] = np.where(test['RUL'] <= w1, 1, 0)
    test['label2'] = test['label1']
    test.loc[test['RUL'] <= w0, 'label2'] = 2
    sequence_length = 100

    def gen_sequence(id_df, seq_length, seq_cols):
        data_matrix = id_df[seq_cols].values
        num_elements = data_matrix.shape[0]
        for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
            yield data_matrix[start:stop, :]
    sensor_cols = ['Altitude', 'Mach_No', 'TRA', 'cycle_norm']
    sequence_cols = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc',
        'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'htBleed', 'W31', 'W32']
    sequence_cols.extend(sensor_cols)

    def gen_labels(id_df, seq_length, label):
        data_matrix = id_df[label].values
        num_elements = data_matrix.shape[0]
        return data_matrix[seq_length:num_elements, :]
    seq_array_test_last = [test[test['unit_nr'] == id][sequence_cols].values[-sequence_length:]
    for id in test['unit_nr'].unique() if len(test[test['unit_nr'] == id]) >=
    sequence_length]
    seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
    y_pred_test = model.predict(seq_array_test_last)
    output = pd.DataFrame(y_pred_test)
    return render('index.html', prediction_text=' ENGINE RUL IS {}'.format(output))



def index(request):
    return render(request,'myapp/index.html')

def predict(request):
    if request.method == 'POST':
        # Load the saved model
        model = keras.models.load_model('C:/Users/Nakitare/Documents/project/AI/models/WS100_dropout_repeat.h5', custom_objects={'r2_keras': r2_keras})

        # Get the uploaded file
        uploaded_file = request.FILES['file']

        # Check that the file type is allowed
        file_ext = uploaded_file.name.split('.')[-1]
        if file_ext not in ALLOWED_FILE_TYPES:
            raise ValidationError(f"Invalid file type. Allowed types are: {', '.join(ALLOWED_FILE_TYPES)}")
        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(uploaded_file.name))
        file_path = uploaded_file.read()
       
        # input_data = np.array(file_path.decode().split(','))
        # Make a prediction using the model
        output = model.predict(file_path)

        # Render the prediction to the user
        context = {'output': output}
        return render(request, 'index.html', context)

    return render(request, 'index.html')


    
           