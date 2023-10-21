#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd 
from dataprep.eda import create_report
import sweetviz as sv


# In[11]:


train=pd.read_csv('https://raw.githubusercontent.com/zerickzhan/Group-20/main/review_train.csv')


# In[12]:


challenge=pd.read_csv('https://raw.githubusercontent.com/zerickzhan/Group-20/main/review_challenge.csv')

# ## EDA

# In[13]:


train


# In[14]:


create_report(challenge)


# In[15]:


analyze_report = sv.analyze(train)
analyze_report.show_html('report.html')


# In[ ]:




