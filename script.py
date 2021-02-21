# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:48:25 2021

@author: ghost
"""

import numpy as np
import pandas as pd

#Training data file
train_file = "./dataset/train.csv"

train_df = pd.read_csv(train_file)
#Separate the target in dataframe
Cost = train_df["Cost"].abs()
Features = train_df.drop(["Cost"], axis=1)

#Retrieve non-numerical feature columns
columns = (Features.dtypes == object)
feature_columns = list(columns[columns].index)

#For plotting relationship of height*width to height
#Create a separate DataFrame for weights, height and width
#from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
#imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
dimensions = Features[["Height", "Width","Weight"]]
missing_weight_index = dimensions[dimensions["Weight"].isnull()].index
drop_na = dimensions.drop(missing_weight_index)

#fill missing values
#import seaborn as sns

#sns.boxplot(drop_na.Height)
#sns.histplot(data=drop_na.Height)
drop_na["Height"] = drop_na["Height"].fillna(drop_na["Height"].mode()[0])
drop_na["Width"] = drop_na["Width"].fillna(drop_na["Width"].mode()[0])
imp_y_dims = drop_na["Weight"]
imp_x_dims = drop_na.drop(["Weight"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    imp_x_dims.to_numpy(),
    imp_y_dims.to_numpy(),
    test_size=0.1,
    random_state=42
    )

import matplotlib.pyplot as plt

plt.scatter(X_train.prod(axis=1),y_train,c="blue")
    
plt.xlabel("Height x Width")
plt.ylabel("Weight")
plt.show()

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
X_standard = scaler.transform(X_train)
pca = PCA(n_components=1).fit(X_standard)
X_train_pca = pca.transform(X_standard)

products = X_train_pca*pca.components_[0]
#plt.scatter(X_train_pca,np.zeros_like(products)+0,c="red")
    
plt.xlabel("Component 1")
plt.ylabel("Weight")
plt.show()

from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(X_train_pca, y_train)
X_test_std = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_std)
pred = regr.predict(X_test_pca)
print(regr.score(X_test_pca, y_test))
x_reshape = X_test_pca.reshape((len(X_test_pca),1))
plt.scatter(X_test_pca,y_test,c="green")
plt.scatter(X_test_pca,pred,c="red")
    
plt.xlabel("Component 1")
plt.ylabel("Weight")
plt.show()

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop="if_binary",sparse=False)
replacement = train_df["Transport"].mode()[0]
train_df["Transport"] = train_df["Transport"].fillna(replacement)
x_transport = pd.DataFrame(ohe.fit_transform(train_df[["Transport"]]), columns=ohe.get_feature_names(["Transport"]))
# x_transport.columns=ohe.get_feature_names(["Transport"])