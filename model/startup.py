#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import h2o
from h2o.automl import H2OAutoML, get_leaderboard
h2o.init(ip="localhost")


df = pd.read_csv (r'founder_V0.3_founder.csv')
df.columns = df.columns.str.replace(' ','_')

#Select features for model
y = "Success"
df[y] = df[y].astype('category')

#Set Factors
x_factor = ["Gender", "Headquarters_Location_"]
df[x_factor] = df[x_factor].astype('category')

#Set Numerics
x_numeric=df.columns[pd.Series(df.columns).str.contains('Number').tolist()].tolist() #Get all columns with "Number" in the name
x_numeric.extend(['Founded_Date']) #add any other necessary columns
df[x_numeric] = df[x_numeric].apply(pd.to_numeric)

#get all features together
x_factor.extend(x_numeric)
x = x_numeric = x_factor
x_y = x+[y]
df=df[x_y]
data_h2o = h2o.H2OFrame(df)


train, test = data_h2o.split_frame(ratios = [.7], seed = 1234)


aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=train)

lb = aml.leaderboard

# Optionally edd extra model information to the leaderboard
lb = get_leaderboard(aml, extra_columns='ALL')

# Print all rows (instead of default 10 rows)
lb.head(rows=lb.nrows)
