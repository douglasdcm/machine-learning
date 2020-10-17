# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:13:14 2019

@author: Douglas
"""

import pandas as pd

base = pd.read_csv('credit_data.csv')
base.loc[base.age < 0, 'age'] = 40.92
               
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

#create the data trainning and test automatically
from sklearn.model_selection import train_test_split
#split 25% of data to test
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)

previsoes = classificador.predict(previsores_teste)

#verify the accuary in prediction of results
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)

#generate the matrix with right and whrng results
matriz = confusion_matrix(classe_teste, previsoes)