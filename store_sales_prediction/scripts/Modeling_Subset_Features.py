
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd 
import xgboost as xgb
import sklearn

import matplotlib.pyplot as plt

for p in [np, pd, xgb, sklearn]:
    print (p.__name__, p.__version__)


# In[3]:


from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import PredefinedSplit, cross_val_score, GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, make_scorer


# In[4]:


RANDOM_STATE = 71


# In[5]:


# Evaluation criterion
def smape(pred, actual):
    """
    pred: a numpy array of predictions
    actual: a numpy array of actual values
    
    for a perfectly predicted zero observation, the smape is defined to be 0. 
    
    """
    
    selector = ~((pred == 0) & (actual == 0))
    numerator = np.abs(pred-actual)
    denom = (np.abs(pred) + np.abs(actual)) / 2
    return 100*np.sum((numerator[selector] / denom[selector])) / pred.shape[0]

smape_scorer = make_scorer(smape, greater_is_better=False)




train = pd.read_csv("../input/train.csv.zip")
test = pd.read_csv("../input/test.csv.zip")
sample_submission = pd.read_csv("../input/sample_submission.csv.zip")





# Convert the date field
train.loc[:,'date'] = pd.to_datetime(train.date)
test.loc[:,'date'] = pd.to_datetime(test.date)


# In[9]:


data = pd.concat([train, test], sort=False).fillna(0)   # test data has id column


# In[10]:


def downcast_dtypes(df):
    '''
        Changes column types in the dataframe: 
                
                `float64` type to `float32`
                `int64`   type to `int32`
    '''
    
    # Select columns to downcast
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype == "int64"]
    
    # Downcast
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols]   = df[int_cols].astype(np.int32)
    
    return df


# In[11]:


data = downcast_dtypes(data)


# In[12]:


# Lag featurizer
class Lag_Featurizer(TransformerMixin):
    def __init__(self, index_col, time_col, value_col, output_col, output_time_index=False, shift=0, freq='1D'):
        self.index_col = index_col
        self.time_col = time_col
        self.value_col = value_col
        self.output_col = output_col
        self.output_time_index=output_time_index
        self.shift = shift
        self.freq = freq
        
    def fit(self, X):                
        pass
    
    def transform(self, X):
        assert isinstance(self.index_col, list)
        
        time_key = pd.Grouper(freq=self.freq)      
        time_index = self.index_col + [time_key]
        resampled = X.groupby(time_index)[self.value_col].sum().reset_index().set_index(self.time_col)
        shifted= resampled.groupby(self.index_col).shift(self.shift, freq=self.freq).drop(self.index_col, axis=1).reset_index().rename(columns={self.value_col:self.output_col})
        merged = pd.merge(X, shifted, how='left',left_on=self.index_col + [self.time_col], right_on=self.index_col + [self.time_col])
        if self.output_time_index:
            return merged.set_index(self.time_col)
        else:
            return merged



data = data.set_index('date')



# 2018-01 ~ 2018-03, a total of 92 days
lag_feature_pipeline = Pipeline(
[
    # lag store, item sales
    
    ('store_item_lag_3m',   Lag_Featurizer(index_col=['store', 'item'],time_col='date',value_col='sales',output_col='sales_3m',   output_time_index=True, shift=98))
#     ('store_item_lag_3m_1', Lag_Featurizer(index_col=['store', 'item'],time_col='date',value_col='sales',output_col='sales_3m_1', output_time_index=True, shift=99)),
#     ('store_item_lag_3m_2', Lag_Featurizer(index_col=['store', 'item'],time_col='date',value_col='sales',output_col='sales_3m_2', output_time_index=True, shift=100)),
#     ('store_item_lag_3m_3', Lag_Featurizer(index_col=['store', 'item'],time_col='date',value_col='sales',output_col='sales_3m_3', output_time_index=True, shift=101)),
#     ('store_item_lag_3m_4', Lag_Featurizer(index_col=['store', 'item'],time_col='date',value_col='sales',output_col='sales_3m_4', output_time_index=True, shift=102)),
#     ('store_item_lag_3m_5', Lag_Featurizer(index_col=['store', 'item'],time_col='date',value_col='sales',output_col='sales_3m_5', output_time_index=True, shift=103)),
#     ('store_item_lag_3m_6', Lag_Featurizer(index_col=['store', 'item'],time_col='date',value_col='sales',output_col='sales_3m_6', output_time_index=True, shift=104)),   
#     ('store_item_lag_4m', Lag_Featurizer(index_col=['store', 'item'],time_col='date',value_col='sales',  output_col='sales_4m',     output_time_index=True, shift=112)),
#     ('store_item_lag_5m', Lag_Featurizer(index_col=['store', 'item'],time_col='date',value_col='sales',  output_col='sales_5m',     output_time_index=True, shift=140)),
#     ('store_item_lag_6m', Lag_Featurizer(index_col=['store', 'item'],time_col='date',value_col='sales',  output_col='sales_6m',     output_time_index=True, shift=168)),
#     ('store_item_lag_9m', Lag_Featurizer(index_col=['store', 'item'],time_col='date',value_col='sales',  output_col='sales_9m',     output_time_index=True, shift=252)),
#     ('store_item_lag_1y', Lag_Featurizer(index_col=['store', 'item'],time_col='date',value_col='sales',  output_col='sales_1y',     output_time_index=True, shift=336))
    
    #lag store sales
#     ('store_lag_3m',   Lag_Featurizer(index_col=['store'],time_col='date',value_col='sales',output_col='store_sales_3m',   output_time_index=True, shift=98)),
#     ('store_lag_3m_1', Lag_Featurizer(index_col=['store'],time_col='date',value_col='sales',output_col='store_sales_3m_1', output_time_index=True, shift=99)),
#     ('store_lag_3m_2', Lag_Featurizer(index_col=['store'],time_col='date',value_col='sales',output_col='store_sales_3m_2', output_time_index=True, shift=100)),
#     ('store_lag_3m_3', Lag_Featurizer(index_col=['store'],time_col='date',value_col='sales',output_col='store_sales_3m_3', output_time_index=True, shift=101)),
#     ('store_lag_3m_4', Lag_Featurizer(index_col=['store'],time_col='date',value_col='sales',output_col='store_sales_3m_4', output_time_index=True, shift=102)),
#     ('store_lag_3m_5', Lag_Featurizer(index_col=['store'],time_col='date',value_col='sales',output_col='store_sales_3m_5', output_time_index=True, shift=103)),
#     ('store_lag_3m_6', Lag_Featurizer(index_col=['store'],time_col='date',value_col='sales',output_col='store_sales_3m_6', output_time_index=True, shift=104)),   
#     ('store_lag_4m', Lag_Featurizer(index_col=['store'],time_col='date',value_col='sales', output_col='store_sales_4m',    output_time_index=True, shift=112)),
#     ('store_lag_5m', Lag_Featurizer(index_col=['store'],time_col='date',value_col='sales', output_col='store_sales_5m',    output_time_index=True, shift=140)),
#     ('store_lag_6m', Lag_Featurizer(index_col=['store'],time_col='date',value_col='sales', output_col='store_sales_6m',    output_time_index=True, shift=168)),
#     ('store_lag_9m', Lag_Featurizer(index_col=['store'],time_col='date',value_col='sales', output_col='store_sales_9m',    output_time_index=True, shift=252)),
#     ('store_lag_1y', Lag_Featurizer(index_col=['store'],time_col='date',value_col='sales', output_col='store_sales_1y',    output_time_index=True, shift=336)),
    
    # lag item sales
#     ('item_lag_3m',   Lag_Featurizer(index_col=['item'],time_col='date',value_col='sales',output_col='item_sales_3m',   output_time_index=True, shift=98)),
#     ('item_lag_3m_1', Lag_Featurizer(index_col=['item'],time_col='date',value_col='sales',output_col='item_sales_3m_1', output_time_index=True, shift=99)),
#     ('item_lag_3m_2', Lag_Featurizer(index_col=['item'],time_col='date',value_col='sales',output_col='item_sales_3m_2', output_time_index=True, shift=100)),
#     ('item_lag_3m_3', Lag_Featurizer(index_col=['item'],time_col='date',value_col='sales',output_col='item_sales_3m_3', output_time_index=True, shift=101)),
#     ('item_lag_3m_4', Lag_Featurizer(index_col=['item'],time_col='date',value_col='sales',output_col='item_sales_3m_4', output_time_index=True, shift=102)),
#     ('item_lag_3m_5', Lag_Featurizer(index_col=['item'],time_col='date',value_col='sales',output_col='item_sales_3m_5', output_time_index=True, shift=103)),
#     ('item_lag_3m_6', Lag_Featurizer(index_col=['item'],time_col='date',value_col='sales',output_col='item_sales_3m_6', output_time_index=True, shift=104)),   
#     ('item_lag_4m', Lag_Featurizer(index_col=['item'],time_col='date',value_col='sales',  output_col='item_sales_4m',   output_time_index=True, shift=112)),
#     ('item_lag_5m', Lag_Featurizer(index_col=['item'],time_col='date',value_col='sales',  output_col='item_sales_5m',   output_time_index=True, shift=140)),
#     ('item_lag_6m', Lag_Featurizer(index_col=['item'],time_col='date',value_col='sales',  output_col='item_sales_6m',   output_time_index=True, shift=168)),
#     ('item_lag_9m', Lag_Featurizer(index_col=['item'],time_col='date',value_col='sales',  output_col='item_sales_9m',   output_time_index=True, shift=252)),
#     ('item_lag_1y', Lag_Featurizer(index_col=['item'],time_col='date',value_col='sales',  output_col='item_sales_1y',   output_time_index=True, shift=336)),

]
)


# In[15]:


get_ipython().run_line_magic('time', 'data = lag_feature_pipeline.transform(data)')


# In[16]:


# drop all rows with nulls. Part of 2013 data is kept since the maximum lag is 336 days. 
data.dropna(inplace=True)
data.loc[:,'weekend'] = ((data.index.weekday == 5) |  (data.index.weekday == 6)) + 0


# In[20]:


cols = [
    
    'sales',

    'sales_3m',  
#     'sales_3m_1',
#     'sales_3m_2',
#     'sales_3m_3',
#     'sales_3m_4',
#     'sales_3m_5',
#     'sales_3m_6',
#     'sales_4m',  
#     'sales_5m',  
#     'sales_6m',  
#     'sales_9m',  
#     'sales_1y',

#     'store_sales_3m',  
#     'store_sales_3m_1',
#     'store_sales_3m_2',
#     'store_sales_3m_3',
#     'store_sales_3m_4',
#     'store_sales_3m_5',
#     'store_sales_3m_6',
#     'store_sales_4m',
#     'store_sales_5m',
#     'store_sales_6m',
#     'store_sales_9m',
#     'store_sales_1y',

#     'item_sales_3m',  
#     'item_sales_3m_1',
#     'item_sales_3m_2',
#     'item_sales_3m_3',
#     'item_sales_3m_4',
#     'item_sales_3m_5',
#     'item_sales_3m_6',
#     'item_sales_4m',  
#     'item_sales_5m',  
#     'item_sales_6m',  
#     'item_sales_9m',  
#     'item_sales_1y',  

    'weekend'
]


# In[22]:


training = data.loc[:'2017-03',cols]
validation_split = np.where((training.index >= pd.Timestamp(2017,1,1)) & (training.index <= pd.Timestamp(2017,3,31)), 0, -1)
print('Number of validation samples = {0}'.format(np.sum(validation_split==0)))


# In[23]:


# Validation data
X_validation = training.loc[validation_split == 0, cols[1:]]
y_validation = training.loc[validation_split == 0, 'sales']
print('Number of training instances = {0:d}'.format(X_validation.shape[0]))
print('Number of features           = {0:d}'.format(X_validation.shape[1]))
print('Date range = {0} to {1}'.format(X_validation.index[0].strftime('%Y-%m-%d'), X_validation.index[-1].strftime('%Y-%m-%d')))
print(X_validation.columns)


# In[24]:


# training matrices
X_training = training.loc[:,cols[1:]]
y_training = training.loc[:,'sales']
print('Number of training instances = {0:d}'.format(X_training.shape[0]))
print('Number of features           = {0:d}'.format(X_training.shape[1]))
print('Date range = {0} to {1}'.format(training.index[0].strftime('%Y-%m-%d'), training.index[-1].strftime('%Y-%m-%d')))
print(X_training.columns)


# In[25]:


testing = data.loc['2018-01':,cols]
X_testing = testing.loc[:,cols[1:]]
y_testing = testing.loc[:,'sales']
print('Number of test instances = {0:d}'.format(X_testing.shape[0]))
print('Number of features       = {0:d}'.format(X_testing.shape[1]))
print('Date range = {0} to {1}'.format(testing.index[0].strftime('%Y-%m-%d'), testing.index[-1].strftime('%Y-%m-%d')))
print(X_testing.columns)


# In[26]:


training_full = data.loc[:'2017-12',cols]
X_training_full = training_full.loc[:,cols[1:]]
y_training_full = training_full.loc[:,'sales']
print('Number of training instances = {0:d}'.format(X_training_full.shape[0]))
print('Number of features           = {0:d}'.format(X_training_full.shape[1]))
print('Date range = {0} to {1}'.format(training_full.index[0].strftime('%Y-%m-%d'), training_full.index[-1].strftime('%Y-%m-%d')))
print(X_training_full.columns)


# #### Random Forest Model

# In[27]:


rf = RandomForestRegressor(random_state=RANDOM_STATE, criterion='mae')


# In[28]:


rf_params = {"n_estimators": np.arange(10, 100, 100),
              "max_depth": np.arange(1, 2, 1)
#               "min_samples_split": np.arange(10,110,10),
#               "min_samples_leaf": np.arange(5,10,1),
#               "max_leaf_nodes": np.arange(5,15,1)
            }


# In[29]:


rf = GridSearchCV(rf, rf_params, scoring='neg_mean_absolute_error', n_jobs=1, cv=PredefinedSplit(validation_split), verbose=50)


# In[ ]:


rf.fit(X_training.values,y_training.values)


# In[89]:


print('Best score = {0:.4f}; Best Parameter = {1}'.format(-rf.best_score_, rf.best_params_))


# In[90]:


# rf_best = rf.best_estimator_
# rf_best.fit(X_training_full, y_training_full)
# pred_rf_best = rf_best.predict(X_testing)
# submission_rf = pd.DataFrame({'Id': sample_submission.id, 'sales': pred_rf_best})
# submission_rf.to_csv('../output/submission_rf.csv', index=False)


