# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 21:17:26 2019

@author: demaomao
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR

data = pd.read_csv(r"C:\DEREK\Learning\ML\05逻辑回归与评分卡\rankingcard.csv", index_col=0)

data.drop_duplicates(inplace=True)
data.index = range(data.shape[0])
#data.info()
#print(data[data["age"] != 0])

data["NumberOfDependents"].fillna(int(data["NumberOfDependents"].mean()), inplace=True)

def fill_missing_rf(X,y,to_fill):

    df = X.copy()
    fill = df.loc[:,to_fill]
    df = pd.concat([df.loc[:,df.columns != to_fill], pd.DataFrame(y)], axis=1)

    Ytrain = fill[fill.notnull()]
    Ytest = fill[fill.isnull()]
    Xtrain = df.iloc[Ytrain.index,:]
    Xtest = df.iloc[Ytest.index, :]

    from sklearn.ensemble import RandomForestRegressor as rfr
    rfr = rfr(n_estimators=100)
    rfr = rfr.fit(Xtrain, Ytrain)
    Ypredict = rfr.predict(Xtest)

    return Ypredict

X = data.iloc[:,1:]
y = data["SeriousDlqin2yrs"]
#print(X.shape)

y_pred = fill_missing_rf(X,y,"MonthlyIncome")


data.loc[data.loc[:,"MonthlyIncome"].isnull(),"MonthlyIncome"] = y_pred

data = data[data["age"]!=0]
data = data[data.loc[:, "NumberOfTimes90DaysLate"] < 90]
data.index = range(data.shape[0])

import imblearn

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X,y = sm.fit_sample(X,y)

n_sample_ = X.shape[0]

pd.Series(y).value_counts()
