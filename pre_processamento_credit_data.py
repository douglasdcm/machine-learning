# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:22:17 2019

@author: Douglas
"""
import pandas as pd
base = pd.read_csv('credit-data.csv')
base.describe() #inform some statistics
base.loc[base['age'] < 0] #find the ages < 0
#to correct...
#delete the column
#base.drop('age', 1, inplace=True)
#delete just the problematic records
#base.drop(base[base.age < 0].index, inplace=True)
#fill the values manually
#fill the values with the average
#base.mean()
#base['age'].mean()
#base['age'][base.age > 0].mean()#get the mean without the problematic values
base.loc[base.age < 0, 'age'] = 40.92#replace the wrong value with the mean


pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]#find the empty ages

previsores = base.iloc[:,1:4].values #get the column from 1 to 3
classe = base.iloc[:,4].values #get the column 4

#fill missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(previsores[:, 0:3])
previsores[:,0:3] = imputer.transform(previsores[:, 0:3])

#scaling the data (put in a same scale)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

