#!/usr/bin/env python
# coding: utf-8

# In[3]:


from dataprep.eda import create_report
import sweetviz as sv


# In[4]:


import pandas as pd 
import pytz
import numpy as np


# In[5]:


train=pd.read_csv('https://raw.githubusercontent.com/zerickzhan/Group-20/main/review_train.csv')

challenge=pd.read_csv('https://raw.githubusercontent.com/zerickzhan/Group-20/main/review_challenge.csv')


# In[6]:


analyze_report = sv.analyze(train)


# In[7]:


analyze_report.show_html('report.html')


# In[8]:


analyze_reportchallenge = sv.analyze(challenge)


# In[9]:


analyze_reportchallenge.show_html('challenge.html')


# In[10]:


count_blank = len(train[(train['user_timezone'].isna() | train['user_timezone'].eq('')) & (train['review_city'].isna() | train['review_city'].eq(''))])

print("Number of rows where both 'review_timestamp' and 'review_city' are blank:", count_blank)


# In[11]:


count_blank2 = len(train[
    (train['user_timezone'].isna() | train['user_timezone'].eq('')) &
    (train['review_city'].isna() | train['review_city'].eq('')) &
    (train['review_coordinates'].isna() | train['review_coordinates'].eq(''))
])

print("Number of rows where 'user_timezone', 'review_city', and 'review_coordinates' are blank:", count_blank)


# In[12]:


debug_selection = len(train[(train['negative_reason_confidence'].isna()) & (train['airline_sentiment'] == 'negative')])
print(debug_selection)


# In[13]:


analyze_report.show_html('report.html')


# From the above EDA extracts we can see: <br>
# 1. The airline sentiment are separated into 3 different kinds with 63% in negative, 21% in neural, and 16% in positive.<br>
# 2. There are 10 negative reasons, and the top 1 accounts for 32% being: Customer service Issue.<br>
# 3. We can see the data is collected from 6 different airline companies, with the top 3 being Emirate (26%), Qatar Airways(20%), and Qantas (19%)<br>
# 
# <br>
# The data is suffering from missing data:<br>
# 
# 1. 92.24% of review coordinates are missing, we this varible is going to cause collinearity issue with review city and we decide to keep review city. <br>
# 
# 2. Contrary to our believe,review city and user time zone do not share all its missing rows. Only 2499 rows are missing for both review city and review time_zone. The inclusion of review coordinate can bring this down to 2418.<br>
# 
# 3. The missing values of negative_reason is caused by the airline_sentiment not being negative. we are going to create a new column called reivew and backfill " positive" or "neural" for the missing values.
# 

# ## Feature Engineering 

# In[14]:


train['review'] = train.apply(lambda row: row['negative_reason'] if not pd.isna(row['negative_reason']) and row['negative_reason'] != '' else row['airline_sentiment'], axis=1)


# Create a new columon the review column. 

# In[15]:


filtered_data = train[train['user_timezone'].isna() & ~train['review_city'].isna()]


# In[16]:


train[train['review_city']=='The other side']


# In[67]:




