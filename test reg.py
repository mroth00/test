import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor 
from sklearn.datasets.california_housing import fetch_california_housing
import pandas as pd

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

train_x=train.ltv
train_y=train.iloc[:,2:10]
train_y['experiencelevel']=train_y['experiencelevel'].astype('category')
train_y['cardtype']=train_y['cardtype'].astype('category')

test_x=test.ltv
test_y=test.iloc[:,2:10]
test_y['experiencelevel']=test_y['experiencelevel'].astype('category')
test_y['cardtype']=test_y['cardtype'].astype('category')


rf = RandomForestRegressor(n_estimators=100)
rf.fit(train_x,train_y)
