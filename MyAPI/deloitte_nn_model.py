from __future__ import print_function
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

import keras
from keras import metrics
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.models import load_model

kc_data_org = pd.read_csv("deloitte_dataset_sheet_with_diseases.csv"); # file location of test data

kc_data_org.head()

kc_data_org.info()

print(kc_data_org.describe())

kc_data = pd.DataFrame(kc_data_org, columns=[
        'age','sex','bmi','children','smoker',
        'region','Alcohol consumer','Diphtheria',
        'Polio','Measles','Hepatitis-B','HIV/AIDS','Thinness',
        'lifetime'])
label_col='lifetime'

print(kc_data.describe())

from sklearn.preprocessing import LabelEncoder

# creating a label encoder
le = LabelEncoder()


# label encoding for sex
# 0 for females and 1 for males
kc_data['sex'] = le.fit_transform(kc_data['sex'])
kc_data['region'] = le.fit_transform(kc_data['region'])
kc_data['smoker'] = le.fit_transform(kc_data['smoker'])
kc_data['Alcohol consumer'] = le.fit_transform(kc_data['Alcohol consumer'])
kc_data['Diphtheria'] = le.fit_transform(kc_data['Diphtheria'])
kc_data['Polio'] = le.fit_transform(kc_data['Polio'])
kc_data['Measles'] = le.fit_transform(kc_data['Measles'])
kc_data['Hepatitis-B'] = le.fit_transform(kc_data['Hepatitis-B'])
kc_data['HIV/AIDS'] = le.fit_transform(kc_data['HIV/AIDS'])
kc_data['Thinness'] = le.fit_transform(kc_data['Thinness'])

kc_data_org.head()

kc_data.head()

def train_validate_test_split(df, train_part=.6, validate_part=.2, test_part=.2, seed=None):
    np.random.seed(seed)
    total_size = train_part + validate_part + test_part
    train_percent = train_part / total_size
    validate_percent = validate_part / total_size
    test_percent = test_part / total_size
    perm = np.random.permutation(df.index)
    m = len(df)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = perm[:train_end]
    validate = perm[train_end:validate_end]
    test = perm[validate_end:]
    return train, validate, test

train_size, valid_size, test_size = (70, 30, 0)
kc_train, kc_valid, kc_test = train_validate_test_split(kc_data, 
                              train_part=train_size, 
                              validate_part=valid_size,
                              test_part=test_size,
                              seed=2017)

kc_y_train = kc_data.loc[kc_train, [label_col]]
kc_x_train = kc_data.loc[kc_train, :].drop(label_col, axis=1)
kc_y_valid = kc_data.loc[kc_valid, [label_col]]
kc_x_valid = kc_data.loc[kc_valid, :].drop(label_col, axis=1)

print('Size of training set: ', len(kc_x_train))
print('Size of validation set: ', len(kc_x_valid))
print('Size of test set: ', len(kc_test), '(not converted)')

def norm_stats(df1, df2):
    dfs = df1.append(df2)
    minimum = np.min(dfs)
    maximum = np.max(dfs)
    mu = np.mean(dfs)
    sigma = np.std(dfs)
    return (minimum, maximum, mu, sigma)

def z_score(col, stats):
    m, M, mu, s = stats
    df = pd.DataFrame()
    for c in col.columns:
        df[c] = (col[c]-mu[c])/s[c]
    return df

stats = norm_stats(kc_x_train, kc_x_valid)
arr_x_train = np.array(z_score(kc_x_train, stats))
arr_y_train = np.array(kc_y_train)
arr_x_valid = np.array(z_score(kc_x_valid, stats))
# arr_x_valid=np.array(kc_x_valid)
arr_y_valid = np.array(kc_y_valid)

print('Training shape:', arr_x_train.shape)
print('Training samples: ', arr_x_train.shape[0])
print('Validation samples: ', arr_x_valid.shape[0])

print(arr_y_train.shape)
arr_y_valid.shape

arr_x_train[:5]

arr_y_train[:5]

def basic_model_1(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(100, activation="tanh", input_shape=(x_size,)))
    t_model.add(Dense(50, activation="relu"))
    t_model.add(Dense(y_size))
    print(t_model.summary())
    t_model.compile(loss='mean_squared_error',
        optimizer=Adam(),
        metrics=[metrics.mae])
    return(t_model)

def basic_model_2(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(100, activation="tanh", input_shape=(x_size,)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(50, activation="relu"))
    t_model.add(Dense(20, activation="relu"))
    t_model.add(Dense(y_size))
    print(t_model.summary())
    t_model.compile(loss='mean_squared_error',
        optimizer=Adam(),
        metrics=[metrics.mae])
    return(t_model)

def basic_model_3(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(80, activation="selu", kernel_initializer='normal', input_shape=(x_size,)))
    t_model.add(Dropout(0.2))
    t_model.add(Dense(40, activation="selu", kernel_initializer='normal', 
        kernel_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(20, activation="selu", kernel_initializer='normal', 
        kernel_regularizer=regularizers.l1_l2(0.01), bias_regularizer=regularizers.l1_l2(0.01)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(10, activation="selu", kernel_initializer='normal'))
    t_model.add(Dropout(0.0))
    t_model.add(Dense(y_size))
    opt = keras.optimizers.Nadam(learning_rate=0.001)
    t_model.compile(
        loss='mean_squared_error',
        optimizer='nadam',
        metrics=[metrics.mae])
    return(t_model)

model = basic_model_3(arr_x_train.shape[1], arr_y_train.shape[1])
model.summary()

epochs = 500
batch_size = 100

print('Epochs: ', epochs)
print('Batch size: ', batch_size)

keras_callbacks = [
    ModelCheckpoint('model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=True, verbose=2),
    ModelCheckpoint('model.{epoch:02d}.hdf5', monitor='val_loss', save_best_only=True, verbose=0),
    TensorBoard(log_dir='model_3', histogram_freq=0, write_graph=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None),
    # EarlyStopping(monitor='val_mean_absolute_error', patience=20, verbose=2)
]

history = model.fit(arr_x_train, arr_y_train,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    verbose=1, # Change it to 2, if wished to observe execution
    validation_data=(arr_x_valid, arr_y_valid),
    callbacks=keras_callbacks)

# import copy 
# mode_copy=copy.deepcopy(model)

model=load_model('model.498.hdf5')

train_score = model.evaluate(arr_x_train, arr_y_train, verbose=0)
valid_score = model.evaluate(arr_x_valid, arr_y_valid, verbose=0)

print('Train MAE: ', round(train_score[1], 4), ', Train Loss: ', round(train_score[0], 4)) 
print('Val MAE: ', round(valid_score[1], 4), ', Val Loss: ', round(valid_score[0], 4))

def plot_hist(h, xsize=6, ysize=10):
    # Prepare plotting
    fig_size = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = [xsize, ysize]
    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True)
    
    # summarize history for MAE
    plt.subplot(211)
    plt.plot(h['mean_absolute_error'])
    plt.plot(h['val_mean_absolute_error'])
    plt.title('Training vs Validation MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # summarize history for loss
    plt.subplot(212)
    plt.plot(h['loss'])
    plt.plot(h['val_loss'])
    plt.title('Training vs Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot it all in IPython (non-interactive)
    plt.draw()
    plt.show()

    return

plot_hist(history.history, xsize=8, ysize=12)

"""Prediction of the Input Data"""

data_new_copy=pd.read_csv("deloitte_dataset_sheet_with_diseases.csv"); # location of the test data

label_col_new='lifetime'
data_new_copy=data_new_copy.drop([label_col_new], axis = 1) 
data_new_copy.head()

model_new=load_model('model.498.hdf5')

from sklearn.preprocessing import LabelEncoder

# creating a label encoder
le_new = LabelEncoder()


# label encoding for sex
# 0 for females and 1 for males
data_new_copy['sex'] = le_new.fit_transform(data_new_copy['sex'])
data_new_copy['region'] = le_new.fit_transform(data_new_copy['region'])
data_new_copy['smoker'] = le_new.fit_transform(data_new_copy['smoker'])
data_new_copy['Alcohol consumer'] = le_new.fit_transform(data_new_copy['Alcohol consumer'])
data_new_copy['Diphtheria'] = le_new.fit_transform(data_new_copy['Diphtheria'])
data_new_copy['Polio'] = le_new.fit_transform(data_new_copy['Polio'])
data_new_copy['Measles'] = le_new.fit_transform(data_new_copy['Measles'])
data_new_copy['Hepatitis-B'] = le_new.fit_transform(data_new_copy['Hepatitis-B'])
data_new_copy['HIV/AIDS'] = le_new.fit_transform(data_new_copy['HIV/AIDS'])
data_new_copy['Thinness'] = le_new.fit_transform(data_new_copy['Thinness'])

print(data_new_copy.head())

age=float(input("What is your age: "))
sex= 1.0 if input("Male or Female ?(M/F)").strip()[0].upper()=='M' else 0.0
bmi=float(input("What is your bmi(Body Mass Index) : "))
children=float(input("How many children do you have ?: "))
smoke= 1.0 if input("Do you smoke ?(Y/N)").strip()[0].upper()=='Y' else 0.0
alcohol= 1.0 if input("Do you drink alcohol ?(Y/N)").strip()[0].upper()=='Y' else 0.0
region= input("Which region do you belong to ?").strip().lower() 
if(region=='northeast'):
  region=0
if(region=='northwest'):
  region=1
if(region=='southeast'):
  region=2
if(region=='southwest'):
  region=3
diptheria= 1.0 if input("Have you suffered from Diphtheria ?(Y/N)").strip()[0].upper()=='Y' else 0.0
polio= 1.0 if input("Have you suffered from Polio  ?(Y/N)").strip()[0].upper()=='Y' else 0.0
measles= 1.0 if input("Have you suffered from Measles  ?(Y/N)").strip()[0].upper()=='Y' else 0.0
hepatitis= 1.0 if input("Have you suffered from Hepatitis-B  ?(Y/N)").strip()[0].upper()=='Y' else 0.0
hiv= 1.0 if input("Have you suffered from HIV/AIDS  ?(Y/N)").strip()[0].upper()=='Y' else 0.0
thinness= 1.0 if input("Have you suffered from Thinness  ?(Y/N)").strip()[0].upper()=='Y' else 0.0

data_inp=pd.DataFrame([[age,sex,bmi,children,smoke, region, alcohol,diptheria,polio,measles,hepatitis,hiv,thinness]],
columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'Alcohol consumer','Diphtheria', 'Polio', 'Measles',
'Hepatitis-B', 'HIV/AIDS', 'Thinness'])
data_new = copy.deepcopy(data_new_copy)
data_new = data_new.append(data_inp,ignore_index=True)
data_x_predict = data_new.loc[:]
def norm_stats_new(dfs):
  minimum = np.min(dfs)
  maximum = np.max(dfs)
  mu = np.mean(dfs)
  sigma = np.std(dfs)
  return (minimum, maximum, mu, sigma)
def z_score(col, stats):
    m, M, mu, s = stats
    df = pd.DataFrame()
    for c in col.columns:
        df[c] = (col[c]-mu[c])/s[c]
    return df
stats_new = norm_stats_new(data_x_predict)
arr_x_predict = np.array(z_score(data_x_predict, stats_new))
data_predict=arr_x_predict[-1:]
trainX_data_new_prdict=model_new.predict(data_predict)
print("The maximum life insurance coverage is: Rs."+str(int(round(trainX_data_new_prdict[-1][0]*10000))))


