#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


# In[2]:


# Updated to newest dataset
DATASET_PATH = os.path.join(".", "training_dataset_V3.csv")
data = pd.read_csv(DATASET_PATH)
data.head()


# In[3]:


'''
# data V2
OLDDATASET_PATH = os.path.join(".", "training_data_cleaned.csv")
olddata = pd.read_csv(OLDDATASET_PATH)
olddata.head()
'''


# In[4]:


# drop unnecessary columns
data = data.drop('Unnamed: 0', axis=1)


# In[5]:


# aggregate columns based on driver ID and keep only most recent entry
data = data.groupby('id_driver').apply(lambda x: x[x['dt'] == x['dt'].max()])


# ## Part 1

# In[6]:


# we convert the column 'most_recent_load_date' into datetime objects in order to calculate the 75th percentile 
most_recent_load_date_datetime = [datetime.datetime.strptime(x, '%Y-%m-%d') for x in data['most_recent_load_date']]
data['most_recent_load_date'] = most_recent_load_date_datetime


# In[7]:


# generate labels based on 75th percentile of 'total_loads' and 'most_recent_load_date'

labels = []

cutoff_date = data['most_recent_load_date'].quantile(0.75)

cutoff_loads = data['total_loads'].quantile(0.75)

for index, row in data.iterrows():
    if(row['total_loads'] >= cutoff_loads and row['most_recent_load_date'] >= cutoff_date):
        labels.append(1)
    else:
        labels.append(0)
        
data['label'] = labels


# In[8]:


data['label'].value_counts()


# ## Part 2

# In[9]:


data = data.drop(['total_loads', 'most_recent_load_date'], axis=1)


# ## Part 3

# In[11]:


data.describe(include='all')


# In[12]:


data.info()


# ### Findings:
# About 25% of the data is missing in features ts_first_approved and days_signup_to_approval, and there is very little data available for the dim_preferred_lanes feature.
# 

# In[13]:


#int/float valued features
data.hist(bins=50, figsize=(20,15))
plt.show()


# In[14]:


#some other plottable categorical features
data['dim_carrier_type'].hist(bins=10, figsize=(5,2.5))
plt.show()


# In[15]:


data['weekday'].hist(bins=50, figsize=(10,2.5))
plt.show()


# In[16]:


data['home_base_state'].hist(bins=50, figsize=(20,5))
plt.show()


# ### Findings:
# 
# We note that most real valued data are highly concentrated in smaller-valued regions with long tails to the right. Year is the only feature that seems more normally distributed. For categorical features, we note that operations are mostly done on work days, and are mostly based in California.

# In[17]:


corr_matrix = data.corr() 
corr_matrix['label'].sort_values(ascending=False)


# In[18]:


#credit: https://towardsdatascience.com/heatmap-basics-with-pythons-seaborn-fb92ea280a6c
fig, ax = plt.subplots(figsize=(12, 10))
# mask
mask = np.triu(np.ones_like(corr_matrix, dtype=np.bool)) 
# adjust mask and df
mask = mask[1:, :-1]
corr = corr_matrix.iloc[1:,:-1].copy()
# color map
cmap = sb.diverging_palette(0, 240, 90, 60, as_cmap=True) 
# plot heatmap
sb.heatmap(corr, mask=mask, annot=True, fmt=".2f",
linewidths=5, cmap=cmap, vmin=-1, vmax=1, cbar_kws={"shrink": .8}, square=True)


# ### Findings:
# 
# - Label is most positively correlated to year, followed by marketplace_loads and marketplace_loads_atlas. In general, it has a positive correlation with all features except days_signup_to_approval.
# - marketplace_loads is highly correlated with marketplace_loads_atlas; same thing between brokerage_loads and brokerage_loads_otr.
# - id_driver is noticeably (positively) correlated with year; days_signup_to_approval is noticeably (negatively) correlated with year and id_driver.

# ## Part 4

# Ideas for feature extraction:
# - many null values in dim_preferred_lanes: change it to a binary variable indicating whether a perferred lane was specified (differentiate between drivers that did and didn't specify)
# 
# Features not needed:
# - ts_signup and ts_first_approved: the period between signup and app
