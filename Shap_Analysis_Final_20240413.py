#!/usr/bin/env python
# coding: utf-8

# # Project Geminae MidPoint Model
# ## Gradient Boosted Regression Model for 3 and 6 month projections
# 
# Tom Gregg
# 
# 2024-02-25

# ## Setting Up The Model

# In[1]:


# Import Basic Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from datetime import datetime


# In[2]:


# Importing Libraries and Packages to perform Boosted Tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from xgboost import XGBRegressor


# In[3]:


# Importing Libraries for Shap Analysis
import shap
shap.initjs()


# In[4]:


# Max Display 
pd.options.display.max_columns = None
pd.options.display.max_rows = None


# ## Importing and Preparing Data

# In[5]:


# Creating our file path for the CSV
file_path = 'GenericWellDataPrepped.csv'
df = pd.read_csv(file_path).copy()


# In[6]:


df.info()


# In[7]:


# Dropping 2020 since the data in this year is thrown off
df = df[df['YearOfDrilling'] != 2020]
# df = df[df['YearOfDrilling'] >= 2017]
# Drop any null 12 month values


# In[8]:


df.dropna(subset=['First12MonthGas_MCFPer1000FT'], inplace=True)


# In[9]:


# df['YearOfDrilling'].value_counts()


# In[10]:


df_cleaned = df.copy()


# In[11]:


# Splitting data into Water, Gas, and Oil 
# Splitting data into 3 month and 6 month
y_w_3 = df_cleaned['First3MonthWater_BBL']
y_g_3 = df_cleaned['First3MonthGas_MCF']
y_o_3 = df_cleaned['First3MonthOil_BBL']
y_w_6 = df_cleaned['First6MonthWater_BBL']
y_g_6 = df_cleaned['First6MonthGas_MCF']
y_o_6 = df_cleaned['First6MonthOil_BBL']
y_w_9 = df_cleaned['First9MonthWater_BBL']
y_g_9 = df_cleaned['First9MonthGas_MCF']
y_o_9 = df_cleaned['First9MonthOil_BBL']
y_w_12 = df_cleaned['First12MonthWater_BBL']
y_g_12 = df_cleaned['First12MonthGas_MCF']
y_o_12 = df_cleaned['First12MonthOil_BBL']
# y_w_36 = df_cleaned['First36MonthWater_BBL']
# y_g_36 = df_cleaned['First36MonthGas_MCFPer1000FT']
# y_o_36 = df_cleaned['First36MonthOil_BBLPer1000FT']
y_w_peak = df_cleaned['PeakWater_BBL']
y_g_peak = df_cleaned['PeakGas_MCF']
y_o_peak = df_cleaned['PeakOil_BBL']
y_w_cum = df_cleaned['CumWater_BBL']
y_g_cum = df_cleaned['CumGas_MCF']
y_o_cum = df_cleaned['CumOil_BBL']


# In[12]:


# Creating X using just the non-production columns
X = df_cleaned.iloc[:, :26]
X = X.drop("Well Index", axis=1)

# Date Cleanup
columns_to_change = ['InitialProductionDate','DrillingStartDate','DrillingCompletionDate']
for col in columns_to_change:
    X[col] = pd.to_datetime(X[col])

# Loop through specific columns and rename
for col in columns_to_change:
    new_name = col + 'Num'
    X.rename(columns={col: new_name}, inplace=True)
    X[new_name] = X[new_name].astype('int64') / 10**9


# Dropping a few unnecessary columns
# X = X.drop('InitialProductionMonth', axis = 1)
X = X.drop('DrillingCompletionDateNum', axis = 1)
X = X.drop('DrillingDuration_DAYS', axis = 1)
# X = X.drop('ProductionMonthsCount', axis = 1)
X = X.drop('YearOfDrilling', axis = 1)
X = X.drop('InitialProductionYear', axis = 1)


# # Dummy Variables for OilTest_Method
# # Use pd.get_dummies to create dummy variables
# dummy_vars = pd.get_dummies(X['OilTest_Method'], prefix='OilTest_Method', drop_first=True)

# # Add the dummy variables as new columns to your DataFrame
# X = pd.concat([X.drop("OilTest_Method", axis=1), dummy_vars], axis=1)

# Converting Objects to Ints
# for col in X.columns:
#     if pd.api.types.is_object_dtype(X[col]):
#         X[col] = X[col].str.replace(',', '')
#         X[col] = X[col].str.replace(' ', '')
#         X[col] = X[col].astype(float)


# In[13]:


X.info()


# In[14]:


X.head()


# In[15]:


# Creating the test and train split using seed 99
# Quite nice how we can just use the exact same X set

# X_train, X_test, y_train_w_3, y_test_w_3 = train_test_split(X, y_w_3, test_size=0.2, random_state=965)

X_train, X_rest1, y_train_w_3, y_rest_w_3_1 = train_test_split(X, y_w_3, test_size=2000, random_state=965)
X_test, X_rest2, y_test_w_3, y_rest_w_3_2 = train_test_split(X_rest1, y_rest_w_3_1, test_size = 1500, random_state=965)
X_calib, X_new, y_calib_w_3, y_new_w_3 = train_test_split(X_rest2, y_rest_w_3_2, test_size = 500, random_state=965)

# X_train, X_test, y_train_g_3, y_test_g_3 = train_test_split(X, y_g_3, test_size=0.2, random_state=965)
X_train, X_rest1, y_train_g_3, y_rest_g_3_1 = train_test_split(X,y_g_3, test_size=2000, random_state=965)
X_test, X_rest2, y_test_g_3, y_rest_g_3_2 = train_test_split(X_rest1, y_rest_g_3_1, test_size = 1500, random_state=965)
X_calib, X_new, y_calib_g_3, y_new_g_3 = train_test_split(X_rest2, y_rest_g_3_2, test_size = 500, random_state=965)

# X_train, X_test, y_train_o_3, y_test_o_3 = train_test_split(X, y_o_3, test_size=0.2, random_state=965)
X_train, X_rest1, y_train_o_3, y_rest_o_3_1 = train_test_split(X,y_o_3, test_size=2000, random_state=965)
X_test, X_rest2, y_test_o_3, y_rest_o_3_2 = train_test_split(X_rest1, y_rest_o_3_1, test_size = 1500, random_state=965)
X_calib, X_new, y_calib_o_3, y_new_o_3 = train_test_split(X_rest2, y_rest_o_3_2, test_size = 500, random_state=965)

# X_train, X_test, y_train_w_6, y_test_w_6 = train_test_split(X, y_w_6, test_size=0.2, random_state=965)
X_train, X_rest1, y_train_w_6, y_rest_w_6_1 = train_test_split(X,y_w_6, test_size=2000, random_state=965)
X_test, X_rest2, y_test_w_6, y_rest_w_6_2 = train_test_split(X_rest1, y_rest_w_6_1, test_size = 1500, random_state=965)
X_calib, X_new, y_calib_w_6, y_new_w_6 = train_test_split(X_rest2, y_rest_w_6_2, test_size = 500, random_state=965)

# X_train, X_test, y_train_g_6, y_test_g_6 = train_test_split(X, y_g_6, test_size=0.2, random_state=965)
X_train, X_rest1, y_train_g_6, y_rest_g_6_1 = train_test_split(X,y_g_6, test_size=2000, random_state=965)
X_test, X_rest2, y_test_g_6, y_rest_g_6_2 = train_test_split(X_rest1, y_rest_g_6_1, test_size = 1500, random_state=965)
X_calib, X_new, y_calib_g_6, y_new_g_6 = train_test_split(X_rest2, y_rest_g_6_2, test_size = 500, random_state=965)

# X_train, X_test, y_train_o_6, y_test_o_6 = train_test_split(X, y_o_6, test_size=0.2, random_state=965)
X_train, X_rest1, y_train_o_6, y_rest_o_6_1 = train_test_split(X,y_o_6, test_size=2000, random_state=965)
X_test, X_rest2, y_test_o_6, y_rest_o_6_2 = train_test_split(X_rest1, y_rest_o_6_1, test_size = 1500, random_state=965)
X_calib, X_new, y_calib_o_6, y_new_o_6 = train_test_split(X_rest2, y_rest_o_6_2, test_size = 500, random_state=965)

# X_train, X_test, y_train_w_9, y_test_w_9 = train_test_split(X, y_w_9, test_size=0.2, random_state=965)
X_train, X_rest1, y_train_w_9, y_rest_w_9_1 = train_test_split(X,y_w_9, test_size=2000, random_state=965)
X_test, X_rest2, y_test_w_9, y_rest_w_9_2 = train_test_split(X_rest1, y_rest_w_9_1, test_size = 1500, random_state=965)
X_calib, X_new, y_calib_w_9, y_new_w_9 = train_test_split(X_rest2, y_rest_w_9_2, test_size = 500, random_state=965)

# X_train, X_test, y_train_g_9, y_test_g_9 = train_test_split(X, y_g_9, test_size=0.2, random_state=965)
X_train, X_rest1, y_train_g_9, y_rest_g_9_1 = train_test_split(X,y_g_9, test_size=2000, random_state=965)
X_test, X_rest2, y_test_g_9, y_rest_g_9_2 = train_test_split(X_rest1, y_rest_g_9_1, test_size = 1500, random_state=965)
X_calib, X_new, y_calib_g_9, y_new_g_9 = train_test_split(X_rest2, y_rest_g_9_2, test_size = 500, random_state=965)

# X_train, X_test, y_train_o_9, y_test_o_9 = train_test_split(X, y_o_9, test_size=0.2, random_state=965)
X_train, X_rest1, y_train_o_9, y_rest_o_9_1 = train_test_split(X,y_o_9, test_size=2000, random_state=965)
X_test, X_rest2, y_test_o_9, y_rest_o_9_2 = train_test_split(X_rest1, y_rest_o_9_1, test_size = 1500, random_state=965)
X_calib, X_new, y_calib_o_9, y_new_o_9 = train_test_split(X_rest2, y_rest_o_9_2, test_size = 500, random_state=965)

# X_train, X_test, y_train_w_12, y_test_w_12 = train_test_split(X, y_w_12, test_size=0.2, random_state=965)
X_train, X_rest1, y_train_w_12, y_rest_w_12_1 = train_test_split(X,y_w_12, test_size=2000, random_state=965)
X_test, X_rest2, y_test_w_12, y_rest_w_12_2 = train_test_split(X_rest1, y_rest_w_12_1, test_size = 1500, random_state=965)
X_calib, X_new, y_calib_w_12, y_new_w_12 = train_test_split(X_rest2, y_rest_w_12_2, test_size = 500, random_state=965)

# X_train, X_test, y_train_g_12, y_test_g_12 = train_test_split(X, y_g_12, test_size=0.2, random_state=965)
X_train, X_rest1, y_train_g_12, y_rest_g_12_1 = train_test_split(X,y_g_12, test_size=2000, random_state=965)
X_test, X_rest2, y_test_g_12, y_rest_g_12_2 = train_test_split(X_rest1, y_rest_g_12_1, test_size = 1500, random_state=965)
X_calib, X_new, y_calib_g_12, y_new_g_12 = train_test_split(X_rest2, y_rest_g_12_2, test_size = 500, random_state=965)

# X_train, X_test, y_train_o_12, y_test_o_12 = train_test_split(X, y_o_12, test_size=0.2, random_state=965)
X_train, X_rest1, y_train_o_12, y_rest_o_12_1 = train_test_split(X,y_o_12, test_size=2000, random_state=965)
X_test, X_rest2, y_test_o_12, y_rest_o_12_2 = train_test_split(X_rest1, y_rest_o_12_1, test_size = 1500, random_state=965)
X_calib, X_new, y_calib_o_12, y_new_o_12 = train_test_split(X_rest2, y_rest_o_12_2, test_size = 500, random_state=965)

# X_train, X_test, y_train_w_peak, y_test_w_peak = train_test_split(X, y_w_peak, test_size=0.2, random_state=965)
X_train, X_rest1, y_train_w_peak, y_rest_w_peak_1 = train_test_split(X,y_w_peak, test_size=2000, random_state=965)
X_test, X_rest2, y_test_w_peak, y_rest_w_peak_2 = train_test_split(X_rest1, y_rest_w_peak_1, test_size = 1500, random_state=965)
X_calib, X_new, y_calib_w_peak, y_new_w_peak = train_test_split(X_rest2, y_rest_w_peak_2, test_size = 500, random_state=965)

# X_train, X_test, y_train_g_peak, y_test_g_peak = train_test_split(X, y_g_peak, test_size=0.2, random_state=965)
X_train, X_rest1, y_train_g_peak, y_rest_g_peak_1 = train_test_split(X,y_g_peak, test_size=2000, random_state=965)
X_test, X_rest2, y_test_g_peak, y_rest_g_peak_2 = train_test_split(X_rest1, y_rest_g_peak_1, test_size = 1500, random_state=965)
X_calib, X_new, y_calib_g_peak, y_new_g_peak = train_test_split(X_rest2, y_rest_g_peak_2, test_size = 500, random_state=965)

# X_train, X_test, y_train_o_peak, y_test_o_peak = train_test_split(X, y_o_peak, test_size=0.2, random_state=965)
X_train, X_rest1, y_train_o_peak, y_rest_o_peak_1 = train_test_split(X,y_o_peak, test_size=2000, random_state=965)
X_test, X_rest2, y_test_o_peak, y_rest_o_peak_2 = train_test_split(X_rest1, y_rest_o_peak_1, test_size = 1500, random_state=965)
X_calib, X_new, y_calib_o_peak, y_new_o_peak = train_test_split(X_rest2, y_rest_o_peak_2, test_size = 500, random_state=965)

# X_train, X_test, y_train_w_cum, y_test_w_cum = train_test_split(X, y_w_cum, test_size=0.2, random_state=965)
X_train, X_rest1, y_train_w_cum, y_rest_w_cum_1 = train_test_split(X,y_w_cum, test_size=2000, random_state=965)
X_test, X_rest2, y_test_w_cum, y_rest_w_cum_2 = train_test_split(X_rest1, y_rest_w_cum_1, test_size = 1500, random_state=965)
X_calib, X_new, y_calib_w_cum, y_new_w_cum = train_test_split(X_rest2, y_rest_w_cum_2, test_size = 500, random_state=965)

# X_train, X_test, y_train_g_cum, y_test_g_cum = train_test_split(X, y_g_cum, test_size=0.2, random_state=965)
X_train, X_rest1, y_train_g_cum, y_rest_g_cum_1 = train_test_split(X,y_g_cum, test_size=2000, random_state=965)
X_test, X_rest2, y_test_g_cum, y_rest_g_cum_2 = train_test_split(X_rest1, y_rest_g_cum_1, test_size = 1500, random_state=965)
X_calib, X_new, y_calib_g_cum, y_new_g_cum = train_test_split(X_rest2, y_rest_g_cum_2, test_size = 500, random_state=965)

# X_train, X_test, y_train_o_cum, y_test_o_cum = train_test_split(X, y_o_cum, test_size=0.2, random_state=965)
X_train, X_rest1, y_train_o_cum, y_rest_o_cum_1 = train_test_split(X,y_o_cum, test_size=2000, random_state=965)
X_test, X_rest2, y_test_o_cum, y_rest_o_cum_2 = train_test_split(X_rest1, y_rest_o_cum_1, test_size = 1500, random_state=965)
X_calib, X_new, y_calib_o_cum, y_new_o_cum = train_test_split(X_rest2, y_rest_o_cum_2, test_size = 500, random_state=965)


# ## Boosted Tree Model
# 
# Scikit-learn reference:
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn-ensemble-gradientboostingregressor

# ### Doing a GridSearchCV
# 

# In[16]:


# # Define the parameter grid
# param_grid = {
#     'learning_rate': [0.01, 0.75, 0.1, 0.25],
#     'n_estimators': [300, 400, 500, 750],
#     'max_depth': [5, 7, 9, 11],
#     'alpha': [0.1, 0.5, 0.75, 0.999]
# }
# gb_mod_t = XGBRegressor(random_state=965)
# grid_search = GridSearchCV(estimator=gb_mod_t, param_grid=param_grid, cv = 2, scoring='r2')
# # Fit the grid search to your data


# In[17]:


# Grid Search, which is worth skipping

# grid_search.fit(X_train, y_train_w_3)


# In[18]:


# Get the best model and its parameters
# best_model = grid_search.best_estimator_
# best_params = grid_search.best_params_

# Print the best parameters and score
# print("Best parameters:", best_params)
# print("Best score:", grid_search.best_score_)


# In[19]:


# pd.DataFrame(grid_search.cv_results_)


# ### Doing a Much faster RandomSearchCV

# In[20]:


# Define distributions for hyperparameters
# from scipy.stats import uniform, randint
# param_dist = {
#     'learning_rate': uniform(0.05, 0.80),
#     'n_estimators': randint(300, 1000),
#     'max_depth': randint(5, 13),
#     'alpha': uniform(0.2, 0.8)
# }


# In[21]:


# # Specify the number of iterations for random search
# n_iter_search = 10

# # Create the RandomizedSearchCV object
# random_search = RandomizedSearchCV(estimator=gb_mod_t, param_distributions=param_dist, n_iter=n_iter_search, cv=5)


# In[22]:


# random_search.fit(X_train, y_train_w_3)


# In[23]:


# best_model = random_search.best_estimator_
# best_score = random_search.best_score_
# # 
# # Print the best parameters and score
# print("Best parameters:", best_params)
# print("Best score:", best_score)


# In[24]:


# pd.DataFrame(random_search.cv_results_)


# ### We will do Water First

# In[25]:


# gb_mod_0 = XGBRegressor(learning_rate=0.1, n_estimators= 300, max_depth = 7, random_state=965, alpha = 0.99)
# gb_mod_0.fit(X_train, y_train_w_3)
# print("XG Boost (default parameters) Train R2: ", gb_mod_0.score(X_train, y_train_w_3))
# print("XG Boost (default parameters) Test R2: ", gb_mod_0.score(X_test, y_test_w_3))


# In[26]:


# gb_mod_1 = XGBRegressor(learning_rate=0.01, n_estimators= 300, max_depth = 7, random_state=965, alpha = 0.99)
# gb_mod_1.fit(X_train, y_train_w_3)
# print("XG Boost (default parameters) Train R2: ", gb_mod_1.score(X_train, y_train_w_3))
# print("XG Boost (default parameters) Test R2: ", gb_mod_1.score(X_test, y_test_w_3))


# In[27]:


# gb_mod_2 = XGBRegressor(learning_rate=1, n_estimators= 300, max_depth = 7, random_state=965, alpha = 0.99)
# gb_mod_2.fit(X_train, y_train_w_3)
# print("XG Boost (default parameters) Train R2: ", gb_mod_2.score(X_train, y_train_w_3))
# print("XG Boost (default parameters) Test R2: ", gb_mod_2.score(X_test, y_test_w_3))


# In[28]:


# gb_mod_3 = XGBRegressor(learning_rate=0.1, n_estimators= 300, max_depth = 9, random_state=965, alpha = 0.99)
# gb_mod_3.fit(X_train, y_train_w_3)
# print("XG Boost (default parameters) Train R2: ", gb_mod_3.score(X_train, y_train_w_3))
# print("XG Boost (default parameters) Test R2: ", gb_mod_3.score(X_test, y_test_w_3))


# In[29]:


# gb_mod_4 = XGBRegressor(learning_rate=0.075, n_estimators= 500, max_depth = 11, random_state=965, alpha = 0.99)
# gb_mod_4.fit(X_train, y_train_w_3)
# print("XG Boost (default parameters) Train R2: ", gb_mod_4.score(X_train, y_train_w_3))
# print("XG Boost (default parameters) Test R2: ", gb_mod_4.score(X_test, y_test_w_3))


# In[30]:


# gb_mod_5 = XGBRegressor(learning_rate=0.075, n_estimators= 500, max_depth = 9, random_state=965, alpha = 0.99)
# gb_mod_5.fit(X_train, y_train_w_3)
# print("XG Boost (default parameters) Train R2: ", gb_mod_5.score(X_train, y_train_w_3))
# print("XG Boost (default parameters) Test R2: ", gb_mod_5.score(X_test, y_test_w_3))


# In[31]:


# gb_mod_6 = XGBRegressor(learning_rate=0.075, n_estimators= 500, max_depth = 9, random_state=965, alpha = 0.99)
# gb_mod_6.fit(X_train, y_train_w_3)
# print("XG Boost (default parameters) Train R2: ", gb_mod_6.score(X_train, y_train_w_3))
# print("XG Boost (default parameters) Test R2: ", gb_mod_6.score(X_test, y_test_w_3))


# ### Fucking Oil Man

# In[32]:


# gb_mod_7 = XGBRegressor(learning_rate=0.075, n_estimators= 400, max_depth = 7, random_state=965, alpha = 0.5)
# gb_mod_7.fit(X_train, y_train_o_3)
# print("XG Boost (default parameters) Train R2: ", gb_mod_7.score(X_train, y_train_o_3))
# print("XG Boost (default parameters) Test R2: ", gb_mod_7.score(X_test, y_test_o_3))


# In[33]:


# gb_mod_8 = XGBRegressor(learning_rate=0.075, n_estimators= 500, max_depth = 10, random_state=965, alpha = 0.5)
# gb_mod_8.fit(X_train, y_train_o_3)
# print("XG Boost (default parameters) Train R2: ", gb_mod_8.score(X_train, y_train_o_3))
# print("XG Boost (default parameters) Test R2: ", gb_mod_8.score(X_test, y_test_o_3))


# In[34]:


# gb_mod_9 = XGBRegressor(learning_rate=0.1, n_estimators= 700, max_depth = 10, random_state=965, alpha = 0.5)
# gb_mod_9.fit(X_train, y_train_o_3)
# print("XG Boost (default parameters) Train R2: ", gb_mod_9.score(X_train, y_train_o_3))
# print("XG Boost (default parameters) Test R2: ", gb_mod_9.score(X_test, y_test_o_3))


# In[35]:


# gb_mod_10 = XGBRegressor(learning_rate=0.1, n_estimators= 500, max_depth = 10, random_state=965, alpha = 0.5)
# gb_mod_10.fit(X_train, y_train_o_3)
# print("XG Boost (default parameters) Train R2: ", gb_mod_10.score(X_train, y_train_o_3))
# print("XG Boost (default parameters) Test R2: ", gb_mod_10.score(X_test, y_test_o_3))


# In[36]:


# One model to run test on 
gb_mod_11 = XGBRegressor(learning_rate=0.035, n_estimators= 450, max_depth = 10, random_state=965, alpha = 0.6)
gb_mod_11.fit(X_train, y_train_o_3)
print("XG Boost (default parameters) Train R2: ", gb_mod_11.score(X_train, y_train_o_3))
print("XG Boost (default parameters) Test R2: ", gb_mod_11.score(X_test, y_test_o_3))


# In[37]:


# Create an empty dictionary to store XGBRegressor instances
gb_mod_10 = {}

# Loop through the range and create/update dictionary entries
for num in range(10):
    var_name = f'gb_mod_10_{num}'  # Construct variable name dynamically
    gb_mod_10[var_name] = XGBRegressor(learning_rate=0.06 + (0.005 * num), n_estimators=500, max_depth=10, random_state=965, alpha=0.5)
    gb_mod_10[var_name].fit(X_train, y_train_o_3)
    print("XG Boost (learning rate = ",0.06 + (0.005 * num),") Train R2: ", gb_mod_10[var_name].score(X_train, y_train_o_3))
    print("XG Boost (learning rate = ",0.06 + (0.005 * num),") Test R2: ", gb_mod_10[var_name].score(X_test, y_test_o_3))


# In[38]:


# Create an empty dictionary to store XGBRegressor instances
gb_mod_10 = {}

# Loop through the range and create/update dictionary entries
for num in range(6):
    var_name = f'gb_mod_10_{num}'  # Construct variable name dynamically
    gb_mod_10[var_name] = XGBRegressor(learning_rate=0.03 + (0.005 * num), n_estimators=500, max_depth=10, random_state=965, alpha=0.5)
    gb_mod_10[var_name].fit(X_train, y_train_o_3)
    print("XG Boost (learning rate = ",0.03 + (0.005 * num),") Train R2: ", gb_mod_10[var_name].score(X_train, y_train_o_3))
    print("XG Boost (learning rate = ",0.03 + (0.005 * num),") Test R2: ", gb_mod_10[var_name].score(X_test, y_test_o_3))


# In[39]:


lr = 0.035 # just optimized for this
md = 10
est = 200
alp = 0.5

# Create an empty dictionary to store XGBRegressor instances
gb_mod_10 = {}

# Loop through the range and create/update dictionary entries
for num in range(10):
    var_name = f'gb_mod_10_{num}'  # Construct variable name dynamically
    gb_mod_10[var_name] = XGBRegressor(learning_rate=lr, n_estimators=est + (50 * num), max_depth=md, random_state=965, alpha=alp)
    gb_mod_10[var_name].fit(X_train, y_train_o_3)
    print("XG Boost ( estimators = ",est + (50 * num),") Train R2: ", gb_mod_10[var_name].score(X_train, y_train_o_3))
    print("XG Boost ( estimators = ",est + (50 * num),") Test R2: ", gb_mod_10[var_name].score(X_test, y_test_o_3))


# In[40]:


lr = 0.035  #optimized
md = 10     #optiimized
est = 450   #optimized
alp = 0.0
#lam = 1

# Create an empty dictionary to store XGBRegressor instances
gb_mod_10 = {}

# Loop through the range and create/update dictionary entries
for num in range(6):
    var_name = f'gb_mod_10_{num}'  # Construct variable name dynamically
    gb_mod_10[var_name] = XGBRegressor(learning_rate=lr, n_estimators=est, max_depth=md, random_state=965, alpha=alp + (0.2*num))
    gb_mod_10[var_name].fit(X_train, y_train_o_3)
    print("XG Boost ( alpha = ", alp + (0.2*num),") Train R2: ", gb_mod_10[var_name].score(X_train, y_train_o_3))
    print("XG Boost ( alpha = ", alp + (0.2*num),") Test R2: ", gb_mod_10[var_name].score(X_test, y_test_o_3))


# In[41]:


lr = 0.035  #optimized
md = 10     #optiimized
est = 450   #optimized
alp = 0.6   #optimized
lam = 0.25

# Create an empty dictionary to store XGBRegressor instances
gb_mod_10 = {}

# Loop through the range and create/update dictionary entries
for num in range(8):
    var_name = f'gb_mod_10_{num}'  # Construct variable name dynamically
    gb_mod_10[var_name] = XGBRegressor(learning_rate=lr, n_estimators=est, max_depth=md, random_state=965, alpha=alp, reg_lambda = lam + (0.25*num))
    gb_mod_10[var_name].fit(X_train, y_train_o_3)
    print("XG Boost ( lambda = ", lam + (0.25*num),") Train R2: ", gb_mod_10[var_name].score(X_train, y_train_o_3))
    print("XG Boost ( lambda = ", lam + (0.25*num),") Test R2: ", gb_mod_10[var_name].score(X_test, y_test_o_3))


# In[42]:


lr = 0.035  #optimized
md = 10     #optiimized
est = 450   #optimized
alp = 0.6   #optimized
lam = 1

# Create an empty dictionary to store XGBRegressor instances
gb_mod_10 = {}

# Loop through the range and create/update dictionary entries
for num in range(1):
    var_name = f'gb_mod_10_{num}'  # Construct variable name dynamically
    gb_mod_10[var_name] = XGBRegressor(learning_rate=lr, n_estimators=est, max_depth=md, random_state=965, alpha=alp)
    gb_mod_10[var_name].fit(X_train, y_train_o_3)
    print("XG Boost ( lambda = ", lam ,") Train R2: ", gb_mod_10[var_name].score(X_train, y_train_o_3))
    print("XG Boost ( lambda = ", lam ,") Test R2: ", gb_mod_10[var_name].score(X_test, y_test_o_3))


# In[43]:


X_train.head()


# In[44]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_pred = gb_mod_11.predict(X_test)
y_test = y_test_o_3

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {round(mae,2)}")
print(f"Mean Squared Error (MSE): {round(mse,2)}")
print(f"Root Mean Squared Error (RMSE): {round(rmse,2)}")
print(f"R-squared (R²): {round(r2,6)}")


# ## We have the optimized values below. Now we need to create a model for all time periods and all elements

# In[45]:


# Great this works

list_of_train_columns=[
y_train_o_3,
y_train_o_6,
y_train_o_9,
y_train_o_12,
y_train_o_cum,
y_train_o_peak,
y_train_w_3,
y_train_w_6,
y_train_w_9,
y_train_w_12,
y_train_w_cum,
y_train_w_peak,
y_train_g_3,
y_train_g_6,
y_train_g_9,
y_train_g_12,
y_train_g_cum,
y_train_g_peak
]

list_of_test_columns = [
y_test_o_3,
y_test_o_6,
y_test_o_9,
y_test_o_12,
y_test_o_cum,
y_test_o_peak,
y_test_w_3,
y_test_w_6,
y_test_w_9,
y_test_w_12,
y_test_w_cum,
y_test_w_peak,
y_test_g_3,
y_test_g_6,
y_test_g_9,
y_test_g_12,
y_test_g_cum,
y_test_g_peak
]

list_of_y_calib_columns = [
y_calib_o_3,
y_calib_o_6,
y_calib_o_9,
y_calib_o_12,
y_calib_o_cum,
y_calib_o_peak,
y_calib_w_3,
y_calib_w_6,
y_calib_w_9,
y_calib_w_12,
y_calib_w_cum,
y_calib_w_peak,
y_calib_g_3,
y_calib_g_6,
y_calib_g_9,
y_calib_g_12,
y_calib_g_cum,
y_calib_g_peak    
]

list_of_y_new_columns = [
y_new_o_3,
y_new_o_6,
y_new_o_9,
y_new_o_12,
y_new_o_cum,
y_new_o_peak,
y_new_w_3,
y_new_w_6,
y_new_w_9,
y_new_w_12,
y_new_w_cum,
y_new_w_peak,
y_new_g_3,
y_new_g_6,
y_new_g_9,
y_new_g_12,
y_new_g_cum,
y_new_g_peak 
]


# In[46]:


display_list_columns=[
'o_3',
'o_6',
'o_9',
'o_12',
'o_cum',
'o_peak',
'w_3',
'w_6',
'w_9',
'w_12',
'w_cum',
'w_peak',
'g_3',
'g_6',
'g_9',
'g_12',
'g_cum',
'g_peak'
]


# In[47]:


# Create a list of tuples by zipping train_list and test_list
data_tuples = []
for i in range(min(len(list_of_train_columns), len(list_of_test_columns))):
    data_tuples.append((list_of_train_columns[i], list_of_test_columns[i]))


# In[48]:


# these them models
lr = 0.035  #optimized
md = 10     #optiimized
est = 450   #optimized
alp = 0.6   #optimized
lam = 1     #optimized

# Create an empty dictionary to store XGBRegressor instances
boosted_models_list = {}
y_pred = {}
# Loop through the range and create/update dictionary entries
for i in range(min(len(list_of_train_columns), len(list_of_test_columns))):
    var_name = f'xgb_mod_{display_list_columns[i]}'  # Construct variable name dynamically
    boosted_models_list[var_name] = XGBRegressor(learning_rate=lr, n_estimators=est, max_depth=md, random_state=965, alpha=alp)
    train_ref = data_tuples[i][0]
    test_ref = data_tuples[i][1]
    boosted_models_list[var_name].fit(X_train, train_ref)
    y_pred[i] = boosted_models_list[var_name].predict(X_test)
    print("XG Boost ( train set = ", display_list_columns[i],") Train R2: ", boosted_models_list[var_name].score(X_train, train_ref))
    print("XG Boost ( test set = ", display_list_columns[i],") Test R2: ", boosted_models_list[var_name].score(X_test, test_ref))


# In[49]:


print("Shape of X_train:", X_train.shape)
print("Shape of train_ref:", train_ref.shape)


# In[50]:


list_of_y_calib_columns[2]


# In[51]:


print(type(boosted_models_list))


# In[52]:


list(boosted_models_list.items())[1][0]


# # Shap Analysis

# In[111]:


shap_values_dict = {}
for i in range(min(len(list_of_train_columns), len(list_of_test_columns))):
    #var_name = f'xgb_mod_{display_list_columns[i]}'
    #model = boosted_models_list[var_name]
    #explainer = shap.Explainer(model.predict, X_test)
    #shap_values = explainer(X_test)
    #shap_values_dict[var_name] = shap_values

    def print_bold_large(text):
        display(HTML(f"<b style='font-size: 20px;'>{text}</b>"))
    print_bold_large(f"SHAP analysis for model: {var_name}") 
    shap.summary_plot(shap_values, X_test,plot_type='bar',plot_size=(12,10))
    shap.summary_plot(shap_values, X_test,plot_size=(12,10))
    
    plots_per_row = 3
    num_features = X_test.shape[1]
    num_rows = (num_features + plots_per_row - 1) // plots_per_row

    fig, axs = plt.subplots(num_rows, plots_per_row, figsize=(17, num_rows*5))

    for i in range(num_features):
        row = i // plots_per_row
        col = i % plots_per_row
    
        ax = axs[row, col] if num_rows > 1 else axs[col]
        shap.dependence_plot(i, shap_values_array, X_test, ax=ax, show=False)
        ax.set_title(X_test.columns[i])
    
    if num_features % plots_per_row != 0:
        for j in range(num_features, num_rows * plots_per_row):
            fig.delaxes(axs.flatten()[j])
        
    plt.tight_layout()
    plt.show()


# # Conformal Predictions

# In[53]:


pip install mapie


# In[54]:


from mapie.regression import MapieRegressor


# In[55]:


for mod in 
y_pred = gb_mod_3.predict(X_test)
mae=mean_absolute_error(y_test_w_3,y_pred)
print(round(mae,2))


# In[93]:


# Need to create the mapie regressor from the gb_mods
# This somehow then needs to look through associated x and ys on the fit
n=0
mapie_reg_list = {}
for guy in display_list_columns:
    var_name = f'mapie_reg_{guy}'
    mapie_reg_list[var_name] = MapieRegressor(estimator=list(boosted_models_list.items())[n][0],cv="prefit")
    mapie_reg_list[var_name].fit(X_calib, list_of_y_calib_columns[n])
    y_pred,y_pis = mapie_reg_list[var_name].predict(X_new,alpha=1/3)
    n = n+1


# In[89]:


list(mapie_reg_list.items())[0][0]


# In[90]:


# Could I somehow loop through the y_calib_x_x as well by just using the placement/index
n = 0
mapie_reg_fit = {}
for i in range(len(display_list_columns)):
    list(mapie_reg_list.items())[n][0].fit(X_calib, list_of_y_calib_columns[n])
    n = n+1


# In[ ]:


# Can loop through to create all of the y_preds
y_pred = gb_mod_3.predict(X_test)
mae=mean_absolute_error(y_test_w_3,y_pred)
print(round(mae,2))


# In[ ]:





# In[ ]:


# Need to loop through the pis as well
y_pred,y_pis = mapie_reg.predict(X_new,alpha=1/3)


# ## Let's make some fucking charts

# In[ ]:


feature_names = X_train.columns
# Extract feature importances from the model
importances = gb_mod_11.feature_importances_
# Sort features and importances in descending order of importance
sorted_idx = importances.argsort()[::-1]
sorted_names = [feature_names[i] for i in sorted_idx][::-1]
sorted_importances = importances[sorted_idx][::-1]

# Create the bar plot
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
plt.barh(sorted_names, sorted_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Feature Name')
plt.title('Feature Importance for Gradient Boosting Model')
plt.xticks(rotation=45, ha='right', fontsize = 8)  # Rotate feature names for better readability
plt.yticks(fontsize = 8)
plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.tree import plot_tree

# Choose the tree index to visualize (between 0 and number of trees - 1)
tree_index = 4  # Change this to the desired tree index

# Extract the tree object from the model
tree = gb_mod_5.estimators_[tree_index]


# In[ ]:


from sklearn.tree import export_graphviz
export_graphviz(
        gb_mod_5,
        out_file="tree.dot",
        feature_names=X_train.columns,
        impurity=False,
        rounded=True,
        filled=True
    )
Source.from_file("tree.dot")


# In[ ]:


dff.describe()


# In[ ]:


df.head()


# In[ ]:


# Sample data (modify with your actual data)
var1 = dff['TrueVerticalDepth_FT']
var2 = dff['MeasuredDepth_FT']

# Create the plot
plt.hist(var1, bins='auto', alpha=0.5, label='Vertical Depth')
plt.hist(var2, bins='auto', alpha=0.5, label='Full Measured Length')
plt.xlabel('Feet')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of Well Depth')
plt.legend()
plt.grid(False)
plt.show()


# In[ ]:


# Sample data (modify with your actual data)
var1 = dff['CumOil_BBL']

# Create the plot
plt.hist(var1, bins='auto', alpha=0.5)
plt.xlabel('Barrels of Oil')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of Oil Production in Barrels')
plt.legend()
plt.grid(False)
plt.show()


# In[ ]:


# Sample data (modify with your actual data)
var1 = dff['ProductionMonthsCount']

# Create the plot
plt.hist(var1, bins='auto', alpha=0.5)
plt.xlabel('Number of Months')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of Production Timeline per Well')
plt.legend()
plt.grid(False)
plt.show()


# In[ ]:


# Create the bar plot
# new imports
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scaler = StandardScaler()
scaler.fit(X_train)
std_x_train = X_train.copy()
std_x_test = X_test.copy()

std_train_array = scaler.transform(std_x_train)
std_test_array = scaler.transform(std_x_test)

std_x_train[:] = std_train_array
std_x_test[:] = std_test_array

# Apply PCA
pca = PCA(n_components=len(X_train.columns))
pca.fit(std_x_train[:])


# Example data: Explained variance ratio for each principal component
explained_variance_ratio = np.array(pca.explained_variance_ratio_)

# Cumulative explained variance
cumulative_explained_variance = pca.explained_variance_ratio_.cumsum()

# Number of components
components = range(1, len(explained_variance_ratio) + 1)

# Creating the plot
plt.figure(figsize=(10, 6))
plt.bar(components, explained_variance_ratio, alpha=0.5, label='Individual explained variance')
plt.plot(components, cumulative_explained_variance, marker='o', linestyle='-', color='r', label='Cumulative explained variance')

plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance')
plt.xticks(components, X_train.columns[:pca.n_components_], rotation=45, fontsize = 8, ha='right')
plt.legend(loc='best')

plt.show()

