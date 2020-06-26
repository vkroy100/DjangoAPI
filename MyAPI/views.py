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
import math
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
				
				model=load_model("/home/ubuntu/DjangoAPI/MyAPI/InsurenceClaim.hdf5")
				p=predict(model,data_inp)
				print(p)
				html = "<html><body>It is now %s.</body></html>" % p
				form={'ans':p}
				return render(request,'ans.html',{'form':form})

				

				
	
	form=ApprovalForm()
				
	return render(request, 'cxform.html', {'form':form})


def predict(model,x_testnp):
	  scaler = MinMaxScaler(feature_range=(0, 1))
	  # only these 4 columns are scaled
	  col_names = ['age', 'bmi', 'children']  
	  features = x_testnp[col_names]
	  features = scaler.fit_transform(features)
	  x_testnp[col_names] = features
	  x_testnp=x_testnp.to_numpy()
	  x_testnp = np.reshape(x_testnp, (x_testnp.shape[0], 1, x_testnp.shape[1]))  
	  val=model.predict(x_testnp)  
	  d=val[-1]
	  return d*10000


