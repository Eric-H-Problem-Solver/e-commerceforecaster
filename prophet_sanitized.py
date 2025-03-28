#!/usr/bin/env python
# coding: utf-8

# codesummary

# In[1]:


#pip install pandas-gbq


# In[1]:


import pandas as pd
import ast
from sklearn.metrics import mean_absolute_error
import datetime
import time
from pyactiveresource.util import to_query
from apscheduler.schedulers.blocking import BlockingScheduler
from pandas import json_normalize
from requests.exceptions import HTTPError
import os
import sys
from dateutil import parser
from sklearn.metrics import mean_squared_error
import itertools
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.model_selection import ParameterGrid
from math import sqrt
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from prophet.plot import plot_plotly, plot_components_plotly
from itertools import product
from tqdm import trange
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from prophet.plot import plot_plotly, plot_components_plotly
from itertools import product
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import xgboost as xgb
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
import warnings
import json
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_squared_error
from math import sqrt
import logging
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import logging
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import ParameterGrid, train_test_split
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import warnings
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_squared_error
from math import sqrt
import logging
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from prophet.plot import plot_plotly, plot_components_plotly
from itertools import product
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_squared_error
from math import sqrt
import logging
from tqdm import tqdm
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from prophet.plot import plot_plotly, plot_components_plotly
from itertools import product
import pandas as pd
from datetime import datetime, timedelta
import pmdarima as pm
import calendar
from pandas.io import gbq
from sklearn.cross_decomposition import PLSRegression


# In[2]:


#pip install gspread pandas oauth2client
import gspread
from oauth2client.service_account import ServiceAccountCredentials


# In[3]:


mod = "Select * FROM panoply.ProphetX"


# In[1]:


mod = "Select * FROM panoply.ProphetX"
df = gbq.read_gbq(mod,  project_id="xxx")


# In[2]:


df


# In[ ]:





# # Manual Loading Method
# mod = '96a2855e-4da8-4559-acc0-c9093d268235_.csv'
# df = pd.read_csv(mod, low_memory=False)
# #df = ogdf

# In[6]:


df['ds'] = pd.to_datetime(df['ds'])
df['launch_date'] = pd.to_datetime(df['launch_date']).dt.tz_localize(None)
ogdf = df


# In[7]:


pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

df['datetime'] = pd.to_datetime(df['ds'], utc=True)
df.set_index('datetime', inplace=True)
df = df[df.index >= '2022-09-01']


# ## Begin Core Arima
# 

# In[8]:


def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year, month)[1])
    return datetime(year, month, day)

# Prediction start and end dates
today_date = datetime.now().date()
start_date = today_date + timedelta(days=1)
end_date = add_months(start_date, 4)


# Monthly SARIMAX

# In[9]:


class Model:
    def __init__(self):
        pass

    def fit(self, training_ds):
        pass

    def predict(self, predict_ds):
        pass 


class BaselineWeightedAverageModel(Model):
    def __init__(self):
        super().__init__()

    def predict(self, predict_ds, predict_date):
        skus = predict_ds['sku'].unique()
        predictions = {sku: 0 for sku in skus}

        dates = [predict_date + datetime.timedelta(days=-x) for x in [90, 60, 30]] + [predict_date]
        weights = [1/6, 1/3, 0.5]

        for weight, (start_date, end_date) in zip(weights, zip(dates[:-1], dates[1:])):
            date_slice = predict_ds[(predict_ds.index.date > start_date) &
                                    (predict_ds.index.date <= end_date)]
            print(weight, start_date, end_date)
            for sku in skus:
                relevant_orders = date_slice[(date_slice['sku'] == sku)]
                predictions[sku] += weight * relevant_orders['quantity'].sum() 
            
        return predictions

def subtract_one_month(date):
    m, y = (date.month-1) % 12, date.year + ((date.month)-2) // 12
    if not m: m = 12
    d = min(date.day, calendar.monthrange(y, m)[1])
    return date.replace(day=d,month=m, year=y)

def slice_quantities_by_column(ds, end_date, column, max_lags=None, slice_by_month=False):
    sliced_quantity_by_sku = {sku: [] for sku in ds[column].unique()}
    n_total = len(ds[ds.index.date <= end_date])
    n_sliced, n_slices = 0, 0

    cur_slice_end_date = end_date
    while n_sliced < n_total:
        if not slice_by_month:
            cur_slice_start_date = cur_slice_end_date + timedelta(days=-30)
        else:
            cur_slice_start_date = subtract_one_month(cur_slice_end_date)
            
        date_slice = ds[(ds.index.date > cur_slice_start_date) &
                                (ds.index.date <= cur_slice_end_date)]
        quantity_by_sku = date_slice.groupby(column)['quantity'].sum().to_dict()
        for sku in sliced_quantity_by_sku:
            sliced_quantity_by_sku[sku].append(quantity_by_sku.get(sku, 0))

        n_sliced += len(date_slice)
        n_slices += 1
        print(f'{n_slices} ({cur_slice_start_date} - {cur_slice_end_date}): {n_sliced}/{n_total}')
        if max_lags and max_lags <= n_slices:
            break
        cur_slice_end_date = cur_slice_start_date

    for k in sliced_quantity_by_sku:
        sliced_quantity_by_sku[k] = sliced_quantity_by_sku[k][::-1]

    return sliced_quantity_by_sku    

class PerSkuARIMAModel(Model):
    def __init__(self, target_period_months, seasonal=False, slice_by_month=False):
        super().__init__()

        self._target_period_months = target_period_months
        self._predictor_by_sku = {}
        self._aic_by_sku = {}  # Store AIC values by SKU
        self._seasonal = seasonal
        self._seasonal_m = 12 if seasonal else 0
        self._slice_by_month = slice_by_month

    def _slice_quantities(self, predict_ds, predict_date, max_lags=None):
        return slice_quantities_by_column(predict_ds, predict_date, column='sku', max_lags=max_lags, slice_by_month=self._slice_by_month)

    def _fit(self, sliced_quantities):
        self._predictor_by_sku = {}

        for sku, series in sliced_quantities.items():
            model = \
                pm.auto_arima(series, error_action='ignore', trace=True,
                suppress_warnings=True, maxiter=5, seasonal=self._seasonal, 
                m=self._seasonal_m, D=0)
            
            self._predictor_by_sku[sku] = model
            self._aic_by_sku[sku] = model.aic()  # Store AIC value

    def predict(self, predict_ds, predict_date):
        sliced_quantities = self._slice_quantities(predict_ds, predict_date)

        self._fit(sliced_quantities)
        
        skus = predict_ds['sku'].unique()
        predictions = {sku: 0 for sku in skus}

        for sku in skus:
            model = self._predictor_by_sku[sku]
            preds = model.predict(n_periods=self._target_period_months+1)
                
            predictions[sku] = preds[:self._target_period_months].sum()
            
        return predictions

class PLSModel(Model):
    def __init__(self, target_period_months, lag_period_months, slice_by_month=False, n_components=1):
        super().__init__()

        self._target_period_months = target_period_months
        self._lag_period_months = lag_period_months
        self._slice_by_month = slice_by_month
        self._n_components = n_components
        self._aic = None  # Store AIC value

    def _slice_quantities(self, predict_ds, predict_date, max_lags=None):
        return slice_quantities_by_column(predict_ds, predict_date, column='sku', max_lags=max_lags, slice_by_month=self._slice_by_month)

    def predict(self, predict_ds, predict_date):
        sliced_quantities = self._slice_quantities(predict_ds, predict_date)

        series_length = len(next(iter(sliced_quantities.values())))
        n = series_length - self._lag_period_months - self._target_period_months
        m = len(sliced_quantities)
        
        mat_predictors = np.zeros((n, m))
        mat_targets = np.zeros((n, m))
        oos_predictors = np.zeros((1, m))

        for idx_sku, (sku, sales_series) in enumerate(sliced_quantities.items()):
            for i in range(n):
                mat_targets[i, idx_sku] = sum(sales_series[i+self._lag_period_months:i+self._lag_period_months+self._target_period_months-1])
                mat_predictors[i, idx_sku] = sum(sales_series[i:i+self._lag_period_months])

            oos_predictors[0, idx_sku] = sum(sales_series[-self._lag_period_months:])                
        
        pls = PLSRegression(n_components=self._n_components)
        pls.fit(mat_predictors, mat_targets)

        pls_oos_predictions = {}
        raw_preds = pls.predict(oos_predictors)
        for idx_sku, sku in enumerate(sliced_quantities):
            pls_oos_predictions[sku] = max(float(raw_preds[0, idx_sku]), 0)
            
        return pls_oos_predictions

class CompositePerSKUArimaPLSModel(Model):
    def __init__(self, target_period_months, slice_by_month=False, n_components=1):
        super().__init__()

        self._target_period_months = target_period_months
        self._slice_by_month = slice_by_month

    def predict(self, predict_ds, predict_date):
        arima = PerSkuARIMAModel(target_period_months=self._target_period_months, slice_by_month=self._slice_by_month)
        pls = PLSModel(target_period_months=self._target_period_months, lag_period_months=self._target_period_months, slice_by_month=self._slice_by_month)

        preds_arima = arima.predict(predict_ds, predict_date)
        preds_pls = pls.predict(predict_ds, predict_date)

        return {k: 0.5 * (v + preds_pls[k]) for k, v in preds_arima.items()}       


# In[10]:


# Create the composite model
fwam = CompositePerSKUArimaPLSModel(target_period_months=4)
preds = fwam.predict(df, predict_date=start_date)

# Create predictions_df DataFrame
predictions_df = pd.DataFrame(list(preds.items()), columns=['SKU', 'Predicted Sales'])

predictions_df['Prediction Date'] = datetime.now().date()

today_date = datetime.now().date()
start_date = today_date # + timedelta(days=1)
fwam = CompositePerSKUArimaPLSModel(target_period_months=4)
preds = fwam.predict(df, predict_date=start_date)


# In[15]:


def eval_predictions(df_eval, prediction_by_sku):
    df_eval_by_sku = df_eval.groupby('sku')
    actual_quanity_by_sku = df_eval_by_sku['quantity'].sum().to_dict()
    average_price_by_sku = (df_eval_by_sku['price'].sum() / df_eval_by_sku['quantity'].sum()).to_dict()

    total_price_val = 0
    total_quantity_val = 0
    total_quantity_delta = 0
    total_price_delta = 0

    for sku, pred in prediction_by_sku.items():
        if not sku in actual_quanity_by_sku:
            continue

         
        quantity_delta = actual_quanity_by_sku[sku] - pred 
        price_delta = quantity_delta * average_price_by_sku[sku]
        
        total_quantity_val += actual_quanity_by_sku[sku]
        total_price_val += actual_quanity_by_sku[sku] * average_price_by_sku[sku]
        total_quantity_delta += abs(quantity_delta)
        total_price_delta += abs(price_delta)
    
    return {
        'total_quantity_delta': total_quantity_delta,
        'total_price_delta': total_price_delta,
        'rel_quantity_delta': total_quantity_delta / total_quantity_val,
        'rel_price_delta': total_price_delta / total_price_val
    }


# In[16]:


preds


# In[18]:


predictions_df = pd.DataFrame(list(preds.items()), columns=['SKU', 'Predicted Sales'])
forecast_period_days = 120

# Add 'Prediction Date' which is today
predictions_df['Prediction Date'] = datetime.now().date()

# Assume 'Start of Forecast Date' is the day after the prediction date
predictions_df['Start of Forecast Date'] = predictions_df['Prediction Date'] + timedelta(days=1)

predictions_df['Forecast End Date'] = predictions_df['Start of Forecast Date'] + timedelta(days=forecast_period_days - 1)

# Calculate 'Average Daily Sales Rate'
predictions_df['Average Daily Sales Rate'] = predictions_df['Predicted Sales'] / forecast_period_days
predictions_df[f'{forecast_period_days}X'] = forecast_period_days * predictions_df['Average Daily Sales Rate']

# View the DataFrame
predictions_df


# In[19]:


predictions_df.to_csv("predictions.csv")


# In[ ]:





# ## End Sarimax Core | Begin Supplementary Files

# Inventory

# In[16]:


df = ogdf
df.dropna(subset=['sku'], inplace=True)

pd.to_datetime(df['launch_date'], utc=True)

end_date = df['ds'].max()
start_date = end_date - pd.DateOffset(months=36)
inventory = df[['sku', 'inventory_quantity']]
inventory_unique = inventory.drop_duplicates(subset=['sku'])

inventory_unique.to_csv('inventory.csv', index=False)


# Drop inventory; trim useless columns and impute where necessary

# In[17]:


df


# In[ ]:





# def fill_category_from_nearest(df):
#     # Iterate over each row in the DataFrame
#     for idx, row in df.iterrows():
#         # Check if 'category' is NaN
#         if pd.isnull(row['category']):
#             # Find rows with the same 'product_type' and a non-NaN 'category'
#             same_type_non_nan = df[(df['product_type'] == row['product_type']) & (~df['category'].isna())]
#             if not same_type_non_nan.empty:
#                 # Pick the first available 'category' from the filtered DataFrame
#                 df.at[idx, 'category'] = same_type_non_nan.iloc[0]['category']
# 
# fill_category_from_nearest(df)
# 
# df

# In[18]:


column_names = df.columns.tolist()

# Print the column names
print(column_names)


# In[19]:


keep = ['ds','sku','line_item_name','quantity','product_type','product_color','cost','price_per_unit', 'category', 'sub_category',  'age','launch_date', 'status','variant_title','vendor']
df = df[keep]
df['launch_date'] = pd.to_datetime(df['launch_date'], errors='coerce')

df = df[~df['sku'].str.contains('TEST', na=False)]
df['age'].fillna('All', inplace=True)

df['category'] = df.groupby('product_type')['category'].transform(lambda x: x.ffill().bfill())

# Impute all 'product_color' with N/A
df['product_color'].fillna('N/A', inplace=True)

# Impute all 'sub_category' with Non-Core
df['sub_category'].fillna('Non-Core', inplace=True)

# Impute all 'published_at' with '1899-01-01' as date value
df['launch_date'].fillna(pd.to_datetime('1899-01-01'), inplace=True)

# Impute all 'vendor' with ?
df['vendor'].fillna('?', inplace=True)

# Impute all 'variant_title' with OS
df['variant_title'].fillna('OS', inplace=True)

# Impute all 'status' with ?
df['status'].fillna('?', inplace=True)
df


# In[20]:


#df['launch_date'].fillna('NA', inplace=True)
#df['launch_date'].replace('NA', pd.Timestamp('1899-01-01'), inplace=True)

nans_per_column = df.isna().sum()
# Print each column name along with the number of NaNs it contains
for column, nan_count in nans_per_column.items():
    print(f"{column}: {nan_count} NaNs")


# In[21]:


nan_product_type_df = df[df['product_type'].isna()]
nan_product_type_df


# In[22]:


columns_with_nans = ['sku', 'product_type', 'product_color', 'cost', 'price_per_unit', 'category', 
                     'age', 'status', 'variant_title', 'vendor']

# Filter rows where any of the specified columns have NaN values
rows_with_nans = df[df[columns_with_nans].isnull().any(axis=1)]
rows_with_nans


# In[23]:


df


# In[24]:


def fill_category_from_nearest(df):
    # Iterate over each row in the DataFrame
    for idx, row in df.iterrows():
        # Check if 'category' is NaN
        if pd.isnull(row['category']):
            # Find rows with the same 'product_type' and a non-NaN 'category'
            same_type_non_nan = df[(df['product_type'] == row['product_type']) & (~df['category'].isna())]
            if not same_type_non_nan.empty:
                # Pick the first available 'category' from the filtered DataFrame
                df.at[idx, 'category'] = same_type_non_nan.iloc[0]['category']

fill_category_from_nearest(df)


# In[25]:


pd.to_datetime(df['launch_date'])

#df.set_index('ds', inplace=True)


# Filtering data
df = df[df['ds'] >= '2022-09-01']

# Remove SKUs containing 'TEST' and fill missing values


# Apply function to fill missing 'category'
fill_category_from_nearest(df)

# Remove duplicate rows based on 'category' and 'product_type' to get unique combinations
unique_combinations = df.drop_duplicates(subset=['category', 'product_type'])

# Additional calculations
unique_skus = df['product_type'].unique()
date_range_days = (df['ds'].max() - df['ds'].min()).days + 1  # +1 to include both start and end dates
total_rows = df.shape[0]
unique_skus_count = df['sku'].nunique()
max_possible_rows = unique_skus_count * date_range_days
total_sales = df['quantity'].sum()

# Print or analyze DataFrame
print("Unique Combinations:", unique_combinations)
print("Total NaN count:", df.isna().sum().sum())
print("Total Sales:", total_sales)

# Display DataFrame (optional)
df


# If data already filled, do not run below code.

# In[26]:


df


# In[27]:


df['launch_date'] = pd.to_datetime(df['launch_date']).dt.tz_localize(None)


# Fill missing categorical values and calculate product_age_days
columns_to_fill = ['launch_date', 'category', 'sub_category', 'product_type', 'age', 'vendor', 'variant_title', 'status', 'product_color']
for column in columns_to_fill:
    df[column] = df.groupby('sku')[column].ffill().bfill()

df['product_age_days'] = (df['ds'] - df['launch_date']).dt.days
df = df[df['ds'] >= df['launch_date']]

# Save and verify the prepared DataFrame
df.to_csv('prophet_fulfilled.csv', index=False)


# In[ ]:





# Sku / cat data Model 2

# In[ ]:





# In[ ]:





# In[ ]:





# ## Sku-Wise Data Application

# In[28]:


print(df.columns)


# In[29]:


end_date = df['ds'].max()
start_date = end_date - pd.DateOffset(months=36)

# Function to calculate sales for specific 30 days periods
def calculate_period_sales(df, start_date, end_date):
    return df[(df['ds'] > start_date) & (df['ds'] <= end_date)]['quantity'].sum()

# Function to calculate the sales for 30D1, 30D2, and 30D3 periods
def calculate_sales_periods(df, last_sale_date):
    period_ends = last_sale_date
    period_starts = period_ends - timedelta(days=30)
    
    sales_30D1 = calculate_period_sales(df, period_starts, period_ends)
    
    period_ends = period_starts
    period_starts = period_ends - timedelta(days=30)
    sales_30D2 = calculate_period_sales(df, period_starts, period_ends)
    
    period_ends = period_starts
    period_starts = period_ends - timedelta(days=30)
    sales_30D3 = calculate_period_sales(df, period_starts, period_ends)
    
    return sales_30D1, sales_30D2, sales_30D3

# Modified weighted average calculation based on the new definition of periods
def weighted_avg_sales(sales_30D1, sales_30D2, sales_30D3):
    weighted_avg = (sales_30D1 * 0.5) + (sales_30D2 * 0.33) + (sales_30D3 * 0.17)
    return weighted_avg

# Aggregate information for each SKU, considering the updated periods
def aggregate_sku_data(group):
    last_sale_date = group['ds'].max()
    sales_30D1, sales_30D2, sales_30D3 = calculate_sales_periods(group, last_sale_date)
    avg_30x = weighted_avg_sales(sales_30D1, sales_30D2, sales_30D3)
    
    return pd.Series({
        '30D1': sales_30D1,
        '30D2': sales_30D2,
        '30D3': sales_30D3,
        '30X': avg_30x,
        'Total Sales': group['quantity'].sum(),
        'Last Sale Date': last_sale_date,
        'Launch Date': group['launch_date'].iloc[0],
        'Category': group['category'].iloc[0],
        'Sub_category': group['sub_category'].iloc[0],
        'Product_type': group['product_type'].iloc[0],
        'Age': group['age'].iloc[0],
        'Status': group['status'].iloc[0],
        'Product_color': group['product_color'].iloc[0],  
        'Cost': group['cost'].iloc[0],                  
        'Price_per_unit': group['price_per_unit'].iloc[0],  
        'Variant_title': group['variant_title'].iloc[0],    
        'Vendor': group['vendor'].iloc[0]              # Assuming you want the first entry
    })

# Use groupby on 'sku' and apply the aggregation function
skus_data = df.groupby('sku').apply(aggregate_sku_data).reset_index()


# In[ ]:


skus_data


# In[31]:


skus_data.to_csv("skuseries.csv")


# In[32]:


import pandas as pd
from datetime import timedelta

# Assuming 'df' is your DataFrame
end_date = df['ds'].max()
start_date = end_date - pd.DateOffset(months=36)

def calculate_period_sales(df, start_date, end_date):
    """ Calculate sales for a specified 30 days period. """
    return df[(df['ds'] > start_date) & (df['ds'] <= end_date)]['quantity'].sum()

def calculate_sales_periods(df, last_sale_date):
    """ Calculate the sales for the 30D1, 30D2, and 30D3 periods. """
    period_ends = last_sale_date
    period_starts = period_ends - timedelta(days=30)
    
    sales_30D1 = calculate_period_sales(df, period_starts, period_ends)
    period_ends = period_starts
    period_starts = period_ends - timedelta(days=30)
    sales_30D2 = calculate_period_sales(df, period_starts, period_ends)
    
    period_ends = period_starts
    period_starts = period_ends - timedelta(days=30)
    sales_30D3 = calculate_period_sales(df, period_starts, period_ends)
    
    return sales_30D1, sales_30D2, sales_30D3

def weighted_avg_sales(sales_30D1, sales_30D2, sales_30D3):
    """ Calculate the weighted average of sales. """
    return (sales_30D1 * 0.5) + (sales_30D2 * 0.33) + (sales_30D3 * 0.17)

def aggregate_sku_data(group):
    """ Aggregate data for each SKU. """
    last_sale_date = group['ds'].max()
    sales_30D1, sales_30D2, sales_30D3 = calculate_sales_periods(group, last_sale_date)
    avg_30x = weighted_avg_sales(sales_30D1, sales_30D2, sales_30D3)
    
    return pd.Series({
        '30D1': sales_30D1,
        '30D2': sales_30D2,
        '30D3': sales_30D3,
        '30X': avg_30x,
        'Total Sales': group['quantity'].sum(),
        'Last Sale Date': last_sale_date,
        'Launch Date': group['launch_date'].iloc[0],
        'Category': group['category'].iloc[0],
        'Sub_category': group['sub_category'].iloc[0],
        'Product_type': group['product_type'].iloc[0],
        'Age': group['age'].iloc[0],
        'Status': group['status'].iloc[0],
        'Product_color': group['product_color'].iloc[0],
        'Cost': group['cost'].iloc[0],
        'Price_per_unit': group['price_per_unit'].iloc[0],
        'Variant_title': group['variant_title'].iloc[0],
        'Vendor': group['vendor'].iloc[0]
    })

# Group by 'sku' and apply the aggregation function
skus_data = df.groupby('sku').apply(aggregate_sku_data).reset_index()

# Create a hybrid column 'Age_ProductType'
skus_data['Age_ProductType'] = skus_data['Age'].astype(str) + '-' + skus_data['Product_type']

# New categorical columns list
categorical_columns = ['Category', 'Sub_category', 'Age_ProductType', 'Product_color', 'Variant_title', 'Vendor']
metrics_columns = ['30D1', '30D2', '30D3', '30X', 'Total Sales']

# Group by the new set of categorical columns and sum the metrics
aggregated_metrics = skus_data.groupby(categorical_columns)[metrics_columns].sum().reset_index()

# Count unique SKUs for each group
sku_counts = skus_data.groupby(categorical_columns)['sku'].nunique().reset_index(name='SKU Count')

# Merge aggregated metrics and SKU counts
aggregated_data = pd.merge(aggregated_metrics, sku_counts, on=categorical_columns, how='left')

# Reset index
aggregated_data.reset_index(drop=True, inplace=True)

# Output to CSV
aggregated_data.to_csv("categories.csv")


# In[33]:


import pandas as pd

# Assuming 'skus_data' is your DataFrame
categorical_columns = ['Category', 'Sub_category', 'Product_type', 'Age', 'Status', 'Launch Date', 'Product_color', 'Variant_title']
metrics_columns = ['30D1', '30D2', '30D3', '30X', 'Total Sales']

# Group by all categorical columns and sum the metrics
aggregated_data = skus_data.groupby(categorical_columns)[metrics_columns].sum().reset_index()

# Calculate the count of unique SKUs for each combination
sku_counts = skus_data.groupby(categorical_columns)['sku'].nunique().reset_index(name='SKU Count')

# Merge the aggregated metrics with the SKU counts
aggregated_data = pd.merge(aggregated_data, sku_counts, on=categorical_columns, how='left')

# Create 'age_size' column by combining 'Age' and 'Variant_title' columns
aggregated_data['age_size'] = aggregated_data['Age'].astype(str) + '-' + aggregated_data['Variant_title'].astype(str)

# Pivot data to view one category against another
pivot_table = pd.pivot_table(aggregated_data, values='Total Sales', 
                             index=['Category', 'Product_color'], 
                             columns=['Product_type', 'age_size'],
                             aggfunc=sum, fill_value=0)

# Reset index if you want it in a flat format for easier export
pivot_table.reset_index(inplace=True)

# Export to CSV
pivot_table.to_csv("pivoter_nocolor_combined.csv", index=False)


# In[34]:


SCOPE = ["https://spreadsheets.google.com/feeds", 
         'https://www.googleapis.com/auth/spreadsheets',
         "https://www.googleapis.com/auth/drive.file",
         "https://www.googleapis.com/auth/drive"]
SERVICE_ACCOUNT_FILE = 'KEY_ID.json'  

credentials = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, SCOPE)
client = gspread.authorize(credentials)

SPREADSHEET_ID = 'googlesheetsurl'
CSV_TO_SHEET_MAP = {
    "skuseries.csv": ("skucore", "A1"),
    "inventory.csv": ("inv", "A1"),
    "categories.csv": ("c_core", "A1"),
    "predictions.csv": ("pcast", "A1"),
    "pivoter_nocolor_combined.csv": ("gsr", "A1")
}

def upload_csv_to_sheet(csv_file, sheet_name, start_cell):
"""
If you're reading this, ask yourself. Why are we still using spreadsheets? Why are we still here?
"""
    df = pd.read_csv(csv_file)

    # Preprocess the DataFrame to replace NaN and infinities and truncate floats
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)  # Replace infinities and fill NaNs
    df = df.applymap(lambda x: round(x, 4) if isinstance(x, float) else x)  # Truncate floats

    # Access the Google Spreadsheet and the specific sheet
    spreadsheet = client.open_by_key(SPREADSHEET_ID)
    try:
        worksheet = spreadsheet.worksheet(sheet_name)
    except gspread.WorksheetNotFound:
        # Create the sheet if it does not exist
        worksheet = spreadsheet.add_worksheet(title=sheet_name, rows="100", cols="20")
    
    # Clear existing data in the sheet to prevent data overlap
    worksheet.clear()
    
    # Convert DataFrame to a list of lists and include the header
    values = [df.columns.values.tolist()] + df.values.tolist()
    
    # Update the sheet with new data, ensuring the correct argument order
    worksheet.update(values, range_name=start_cell)

def main():
    # Iterate over the CSV files and their target Google Sheets and cells
    for csv_file, (sheet_name, start_cell) in CSV_TO_SHEET_MAP.items():
        if os.path.exists(csv_file):
            print(f"Uploading {csv_file} to {sheet_name} starting at {start_cell}...")
            upload_csv_to_sheet(csv_file, sheet_name, start_cell)
        else:
            print(f"File {csv_file} not found.")

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:


# Spreadsheet URL ID and desired CSVs with sheet and starting cell
# Product Performance Analysis
SPREADSHEET_ID = 'URL FOR ANALYSIS'
CSV_TO_SHEET_MAP = {
    "skuseries.csv": ("skucore", "A1"),
    "inventory.csv": ("inv", "A1"),
    "categories.csv": ("c_core", "A1"),
    "predictions.csv": ("pcast", "A1"),
    "pivoter_nocolors_synth.csv": ("gsr", "A1")
}
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:




