#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd 
from dataprep.eda import create_report
import sweetviz as sv


# In[11]:


challenge=pd.read_csv("E:/USyd/QBUS6850 machine learning for business - Semester2/Assignment/review_challenge.csv")


# In[12]:


train= pd.read_csv("E:/USyd/QBUS6850 machine learning for business - Semester2/Assignment/review_train.csv")


# ## EDA

# In[13]:


train


# In[14]:


create_report(challenge)


# In[15]:


analyze_report = sv.analyze(train)
analyze_report.show_html('report.html')


# In[ ]:




