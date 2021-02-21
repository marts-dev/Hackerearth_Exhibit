# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 00:01:06 2021

@author: ghost
"""

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('labelencode', LabelEncoder())
])

class TransportTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('Init TransportTransformer')
        self.ohe = OneHotEncoder(sparse=False)
        
    def fit(self, x, y=None):
        x["Transport"].fillna(x.Transport.mode()[0], inplace=True)
        x["Transport"] = x["Transport"].map(lambda c: c[0])
        self.ohe.fit( x["Transport"])
        return self

    def transform(self, x):
        x["Transport"].fillna(x.Transport.mode()[0], inplace=True)

        x_transport = pd.DataFrame(self.ohe.transform(x[["Transport"]]), columns=self.ohe.get_feature_names(["Transport"]))
        x.drop(["Transport"], axis=1, inplace=True)
        x = pd.concat([x, x_transport], axis=1)
    
        return x

class LocationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('Init LocationTransformer')
        
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        
        x["State"] = x["Customer Location"].map(lambda x:str(x).split()[-2])
        x.drop(["Customer Location"], axis=1, inplace=True)
    
        return x

class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('Init DateTransformer')
        
    def fit(self, x, y=None):
        x["Scheduled Date"] = pd.to_datetime(x["Scheduled Date"], format="%m/%d/%y")
        x["Delivery Date"] = pd.to_datetime(x["Delivery Date"], format="%m/%d/%y")
        return self

    def transform(self, x):
        x["ScheduleDiff"] = (x["Delivery Date"]-x["Scheduled Date"]).map(lambda x: str(x).split()[0])
        x["ScheduleDiff"] = pd.to_numeric(x["ScheduleDiff"])
        x.drop(["Delivery Date", "Scheduled Date"], axis=1, inplace=True)
    
        return x

class CostTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('Init CostTransformer')
        
    def fit(self, x, y=None):
        if y is not None:
            y = np.log1p(abs(y))
        return self

    def transform(self, x):
    
        return x