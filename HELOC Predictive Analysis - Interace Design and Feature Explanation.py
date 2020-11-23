# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 16:08:39 2019

@author: hp
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:35:40 2019

@author: Siyuan Feng
"""
import streamlit as st
import streamlit.components.v1 as components
import pickle

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import lime
import lime.lime_tabular


# load dataset
data = pd.read_csv('heloc_dataset_v1.csv',header = 0)

# change target variable to labels
data['RiskPerformance'] = data['RiskPerformance'].replace("Bad",0)
data['RiskPerformance'] = data['RiskPerformance'].replace("Good",1)


# deal with special values
# - remove observations without any record
data = data[data.ExternalRiskEstimate!=-9]

# - impute median for observations lack record for observations lacking certain records
for i in data.columns:
    data[i] = data[i].replace(-7,np.nan)
    data[i] = data[i].replace(-8,np.nan)

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
data = pd.DataFrame(imputer.fit_transform(data),columns= data.columns)
data['RiskPerformance'] = data['RiskPerformance'].replace(0,"Bad")
data['RiskPerformance'] = data['RiskPerformance'].replace(1,"Good")

# convert categorical variables to dummy variables
data = pd.get_dummies(data, columns=['MaxDelq2PublicRecLast12M'], drop_first=False)
data = pd.get_dummies(data, columns=['MaxDelqEver'], drop_first=False)  

# remove obvious outliers
data = data[(data['NumInqLast6M']<60)&(data['NumInqLast6Mexcl7days']<60)&\
      (data['NetFractionRevolvingBurden']<200)&(data['NetFractionInstallBurden']<400)]

# load X and y
X = data.iloc[:,1:]
y = data.iloc[:,0]
le = LabelEncoder()
le.fit(y)
y = le.transform(y)


# split training set and test set
np.random.seed(1)
X_train_original, X_test_original, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# feature scaling
scaler = StandardScaler()
scaler.fit(X_train_original)
X_train = scaler.transform(X_train_original)
X_test = scaler.transform(X_test_original)

# train the prediction model
rf = RandomForestClassifier(random_state=1,n_estimators=98)
rf.fit(X_train, y_train)


# Title
st.title('HELOC Risk Prediction')
st.subheader('Information Collection')

# Checkbox to show the whole dataset
if st.checkbox('Show Test data'):
    test_data = X_test_original
    st.dataframe(test_data)

st.write('')

st.write('Choose an observation in the test data to get the test result:')

# Selectbox to choose the observation
number = st.number_input('Input a row number')
test = X_test_original.iloc[int(number),:].values


res = rf.predict([test])

st.write('_Model: Random Forests_')
st.write('_Accuracy: 72%_')
st.write('_Rate of successfully identifying clients with high risk: 72%_')

st.write('')
st.subheader('Prediction Result')
st.write('')

if res == 0:
    st.markdown('**_Prediction: Bad_**')
else:
    st.markdown('**_Prediction: Good_**')


# explaination with LME
feature_names = data.columns[1:]
labels = data.iloc[:,0]
class_names = le.classes_
features = data.iloc[:,1:]
categorical_names = {}
categorical_features = list(range(21,37))
for feature in categorical_features:
    categorical_names[feature] = np.array([0,1])
    
predict_fn = lambda x: rf.predict_proba(scaler.transform(x)).astype(float)

explainer = lime.lime_tabular.LimeTabularExplainer(X_train_original.values ,feature_names = feature_names,class_names=class_names,
                                                   categorical_features=categorical_features, 
                                                   categorical_names=categorical_names, kernel_width=3)
exp = explainer.explain_instance(test, predict_fn, num_features=5)

st.subheader('Feature Explanation')
st.write(exp.as_pyplot_figure())


# slider
ExternalRiskEstimate = st.sidebar.slider("External Risk Estimate",0,100,0)

MSinceOldestTradeOpen=st.sidebar.slider("Months Since Oldest Trade Open",0,810,0)
    
MSinceMostRecentTradeOpen=st.sidebar.slider("Months Since Most Recent Trade Open",0,400,0)
    
AverageMInFile=st.sidebar.slider("Average Months in File",0,400,0)
   
NumSatisfactoryTrades= st.sidebar.slider("Satisfactory Trades Number",0,80,0)
    
NumTrades60Ever2DerogPubRec=st.sidebar.slider("Trades 60+ Ever",0,20,0)
    
NumTrades90Ever2DerogPubRec=st.sidebar.slider("Trades 90+ Ever",0,20,0)
    
NumTotalTrades=st.sidebar.slider("Total Number of Credit Accounts",0,110,0)

PercentInstallTrades=st.sidebar.slider('Percent Installment Trades',0,110,0)
    
NumTradesOpeninLast12M=st.sidebar.slider("Number of Trades Open in Last 12 Months",0,20,0)

PercentTradesNeverDelq=st.sidebar.slider("Percent Trades Never Delinquent",0,100,0)
   
MSinceMostRecentDelq=st.sidebar.slider("Months Since Most Recent Delinquent",0,90,0)

MSinceMostRecentInqexcl7days=st.sidebar.slider("Months Since Most Recent Inquiry excl 7days",0,30,0)
    
NumInqLast6M=st.sidebar.slider("Number of Inquiry Last 6 Months",0,70,0)
    
NumInqLast6Mexcl7days=st.sidebar.slider("Number of Inquiry Last 6 Months excl 7days",0,70,0)

NetFractionRevolvingBurden=st.sidebar.slider("Net Fraction Revolving Burden",0,240,0)
    
NetFractionInstallBurden=st.sidebar.slider('Net Fraction Install Burden',0,480,0)

NumRevolvingTradesWBalance=st.sidebar.slider("Revolving Trades with Balance",0,40,0)

NumInstallTradesWBalance=st.sidebar.slider('Installment Trades with Balance Number',0,25,0)

NumBank2NatlTradesWHighUtilization=st.sidebar.slider("Bank/National Trades with high utilization ratio Number",0,20,0)

PercentTradesWBalance=st.sidebar.slider("Percent Trades with Balance",0,100,0)

MaxDelq2PublicRecLast12M0=st.sidebar.slider("Max Delq/Public Records Last 12 Months",0,1,0)
    
MaxDelqEver=st.sidebar.slider("Max Delinquency Ever",0,8,0)

#Using trained model to predict through the arrary of above features

res = rf.predict(np.array([ExternalRiskEstimate,MSinceOldestTradeOpen,MSinceMostRecentTradeOpen,MSinceMostRecentTradeOpen, NumSatisfactoryTrades, NumTrades60Ever2DerogPubRec, NumTrades90Ever2DerogPubRec, NumTotalTrades, PercentInstallTrades, NumTradesOpeninLast12M, PercentTradesNeverDelq, MSinceMostRecentDelq, MaxDelq2PublicRecLast12M, MaxDelqEver, MSinceMostRecentInqexcl7days, NumInqLast6M, NumInqLast6Mexcl7days, NetFractionRevolvingBurden, NetFractionInstallBurden, NumRevolvingTradesWBalance, NumInstallTradesWBalance,NumBank2NatlTradesWHighUtilization,PercentTradesWBalance]).reshape(1,-1))

if res == 0:
     st.markdown('**Prediction: Bad**')
else:
     st.markdown('**Prediction: Good**')


# explaination with LME
feature_names = data.columns[1:]
labels = data.iloc[:,0]
class_names = le.classes_
features = data.iloc[:,1:]
categorical_names = {}
categorical_features = list(range(21,37))
for feature in categorical_features:
    categorical_names[feature] = np.array([0,1])
    
predict_fn = lambda x: rf.predict_proba(scaler.transform(x)).astype(float)

explainer = lime.lime_tabular.LimeTabularExplainer(X_train_original.values ,feature_names = feature_names,class_names=class_names,
                                                   categorical_features=categorical_features, 
                                                   categorical_names=categorical_names, kernel_width=3)
exp = explainer.explain_instance(test, predict_fn, num_features=5)
st.write(exp.as_pyplot_figure())
    