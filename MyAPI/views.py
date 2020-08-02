from django.shortcuts import render
from rest_framework import viewsets
from django.core import serializers
from rest_framework.response import Response
from rest_framework import status
from . forms import ApprovalForm
from django.http import HttpResponse
from django.http import JsonResponse
from django.contrib import messages
from . models import approvals
import pickle
import tensorflow as tf
from tensorflow.keras import backend as K
import joblib
import numpy as np
from sklearn import preprocessing
import pandas as pd
from collections import defaultdict, Counter
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import math
import copy
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau




def cxcontact(request):
	if request.method=='POST':
		form=ApprovalForm(request.POST)
		if form.is_valid():
				Firstname = form.cleaned_data['firstname']
				Lastname = form.cleaned_data['lastname']
				age = float(form.cleaned_data['age'])
				sex =1.0 if form.cleaned_data['gender']=='Male' else 0.0
				bmi = float(form.cleaned_data['bmi'])
				children = float(form.cleaned_data['children'])
				smoke = 1.0 if form.cleaned_data['smoker']=='Yes' else 0.0
				alcohol =1.0 if form.cleaned_data['alcoholConsumer']=='Yes' else 0.0
				region = form.cleaned_data['region']
				if(region=='northeast'):
  					region=0.0
				if(region=='northwest'):
				  	region=1.0
				if(region=='southeast'):
  					region=2.0
				if(region=='southwest'):
  					region=3.0
				diptheria = 1.0 if form.cleaned_data['diphtheria'] == 'Yes' else 0.0
				polio = 1.0 if form.cleaned_data['polio'] == 'Yes'else 0.0
				measles = 1.0 if form.cleaned_data['measles'] == 'Yes' else 0.0
				hepatitis = 1.0 if form.cleaned_data['hepatitis'] == 'Yes' else 0.0
				hiv = 1.0 if form.cleaned_data['hiv_aids'] == 'Yes' else 0.0
				thinness = 1.0 if form.cleaned_data['thinness'] == 'Yes' else 0.0
				data_inp=pd.DataFrame([[age,sex,bmi,children,smoke,region,alcohol,diptheria,polio,measles,hepatitis,hiv,thinness]],
				columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'Alcohol consumer','Diphtheria', 'Polio', 'Measles',
						 'Hepatitis-B', 'HIV/AIDS', 'Thinness'])
				data_new_copy=pd.read_csv("/home/ubuntu/DjangoAPI/MyAPI/deloitte_dataset_sheet_with_diseases.csv");
				label_col_new='lifetime'
				data_new_copy=data_new_copy.drop([label_col_new], axis = 1) 
				
				model_new=load_model("/home/ubuntu/DjangoAPI/MyAPI/model.hdf5")
				le_new = LabelEncoder()

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
				data_new = copy.deepcopy(data_new_copy)
				data_new = data_new.append(data_inp,ignore_index=True)
				data_x_predict = data_new.loc[:]
				stats_new = norm_stats_new(data_x_predict)
				arr_x_predict = np.array(z_score(data_x_predict, stats_new))
				data_predict=arr_x_predict[-1:]
				trainX_data_new_prdict=model_new.predict(data_predict)
				p=float(round(trainX_data_new_prdict[-1][0]*1000))
				html = "<html><body>It is now %s.</body></html>" % p
				form={'ans':p}
				return render(request,'ans.html',{'form':form})
				
	
	form=ApprovalForm()
				
	return render(request, 'cxform.html', {'form':form})

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
