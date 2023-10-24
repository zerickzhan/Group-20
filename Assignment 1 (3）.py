#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dataprep.eda import create_report
import sweetviz as sv


# In[2]:


import pandas as pd 
import pytz
import numpy as np


# In[3]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torchtext.data.utils import get_tokenizer

import torch
from torch import nn 
import torch.nn.functional as F
from torch import utils

torch.manual_seed(0)
np.random.seed(0)

import warnings
warnings.filterwarnings("ignore")


# In[4]:


train=pd.read_csv('https://raw.githubusercontent.com/zerickzhan/Group-20/main/review_train.csv')
challenge=pd.read_csv('https://raw.githubusercontent.com/zerickzhan/Group-20/main/review_challenge.csv')


# In[5]:


analyze_report = sv.analyze(train)


# In[6]:


analyze_report.show_html('report.html')


# In[7]:


analyze_reportchallenge = sv.analyze(challenge)


# In[8]:


analyze_reportchallenge.show_html('challenge.html')


# In[9]:


count_blank = len(train[(train['user_timezone'].isna() | train['user_timezone'].eq('')) & (train['review_city'].isna() | train['review_city'].eq(''))])

print("Number of rows where both 'review_timestamp' and 'review_city' are blank:", count_blank)


# In[10]:


count_blank2 = len(train[
    (train['user_timezone'].isna() | train['user_timezone'].eq('')) &
    (train['review_city'].isna() | train['review_city'].eq('')) &
    (train['review_coordinates'].isna() | train['review_coordinates'].eq(''))
])

print("Number of rows where 'user_timezone', 'review_city', and 'review_coordinates' are blank:", count_blank)


# In[11]:


debug_selection = len(train[(train['negative_reason_confidence'].isna()) & (train['airline_sentiment'] == 'negative')])
print(debug_selection)


# In[12]:


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

# In[13]:


train['review'] = train.apply(lambda row: row['negative_reason'] if not pd.isna(row['negative_reason']) and row['negative_reason'] != '' else row['airline_sentiment'], axis=1)


# Create a new columon the review column. 

# In[14]:


filtered_data = train[train['user_timezone'].isna() & ~train['review_city'].isna()]


# In[15]:


# preprocessing data
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder() 
target = le.fit_transform(train['airline_sentiment']) # convert target into integers
train['airline_sentiment'] = target
print(le.classes_) # this shows which index maps to which class


# In[16]:


np.random.seed(511365)

index = list(range(train.shape[0])) # an list of indices
np.random.shuffle(index) # shuffle the index in-place

p_val = 0.2
p_test = 0.2
N_test = int(train.shape[0] * p_test)
N_val = int(train.shape[0] * p_val)


# get training, val and test sets
test_data = train.iloc[ index[:N_test] ,:]
val_data = train.iloc[ index[N_test: (N_test+N_val)], :]
train_data = train.iloc[ index[(N_test+N_val):], :]

print(test_data.shape)
print(val_data.shape)
print(train_data.shape)


# In[17]:


# define our own torch dataset
# for a torch dataset, we need to define two functions: 
#     __len__: return the length of dataset
#     __getitem__: given a index (integer), return the corresponding sample, both y and X

class SpamDataset(utils.data.Dataset):
    def __init__(self, myData):
        """
        myData should be a dataframe object containing both y (first col) and X (second col)
        """
        super().__init__()
        self.data = myData
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        return (self.data.iloc[idx,2], self.data.iloc[idx,7]) # (target, text)


# In[18]:


# now we can build our torch dataset 
train_torch = SpamDataset(train_data)
val_torch = SpamDataset(val_data)
test_torch = SpamDataset(test_data)


# In[19]:


# check
train_torch.__getitem__(3)


# In[20]:


tokenizer = get_tokenizer('basic_english')


# In[21]:


from torchtext.vocab import build_vocab_from_iterator

# ===== Build vocabulary =====
# an unknown token is added for all unknown words outside the documents
# you may specify the min_freq to filter out infrequent words
vocabulary = build_vocab_from_iterator(
    [tokenizer(msg) for msg in train['review_text']],
    specials=["<unk>"],
    min_freq = 3, # filter out all words that appear less than three times
)
# Set to avoid errors with unknown words
vocabulary.set_default_index(vocabulary["<unk>"])


# In[22]:


len(vocabulary)


# In[23]:


# The vocab object maps a word to an idx (an integer)
print(vocabulary['better'])
print(vocabulary['sun'])
print(vocabulary['iertuei']) # something not in vocab will be mapped to default_index = 0


# In[29]:


# define a function that converts a document into tokens (represented by index)
def doc_tokenizer(doc):
    return torch.tensor([vocabulary[token] for token in tokenizer(doc)], dtype=torch.long)


# In[24]:


# ========= Step 2 ==============
# Notice in a corpus, each document can have different size. Thus, we usually pad zeros to the maximum length of document.
# Alternatively, you can concat all documents into a long vector 
# and the starting index of each document is identified in the variable called offsets.

def collate_batch(batch):
    
    target_list, text_list, offsets = [], [], [0] # initalize the first offset to be 0 here
        
    # loop through all samples in batch
    for idx in range(len(batch)):
        
        _label = batch[idx][0]
        _text = batch[idx][1]
        
        target_list.append( _label )
        tokens = doc_tokenizer( _text )
        text_list.append(tokens)
        
        # ====== wrong ======
        #if idx == 0:
        #    offsets.append(0)  # the first document starts from idx 0
        #else:
        #    offsets.append(offsets[-1] + tokens.size(0)) # the next document starts from (offsets[-1] + tokens.size(0))
        # ===================
        
        offsets.append(offsets[-1] + tokens.size(0))
    
    offsets = offsets[:-1] # remove the last entry
    
    # convert to torch tensor
    target_list = torch.tensor(target_list, dtype=torch.int64)
    offsets = torch.tensor(offsets)
    text_list = torch.cat(text_list) # concat into a long vector
    
    return target_list, text_list, offsets


# In[28]:


torch.manual_seed(0)

batchSize = 8
train_loader = utils.data.DataLoader(train_torch, batch_size=batchSize, shuffle=True, collate_fn=collate_batch)
val_loader = utils.data.DataLoader(val_torch, batch_size=batchSize, shuffle=True, collate_fn=collate_batch)
test_loader = utils.data.DataLoader(test_torch, batch_size=batchSize, shuffle=False, collate_fn=collate_batch)


# In[30]:


list(train_loader)[0]


# Model building

# In[32]:


# ====== Step 1 ========= 
class SpamClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode='mean') # embedding layer
        self.Linear1 = nn.Linear(embed_dim, 1)
        self.Dropout = nn.Dropout(p=0.1)
    
    def forward(self, text, offsets):
        # note we need offsets to indicate which document we have
        out = self.embedding(text, offsets)
        out = self.Dropout(out)
        out = self.Linear1(out)
        return out
        # for the last layer, we don't apply activation because we can use BCEWithLogitsLoss to combine sigmoid with BCELoss
        
# model initalization
embed_dim = 8
model = SpamClassifier(len(vocabulary), embed_dim)


# In[33]:


# ======= Step 2 ==========
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


# In[34]:


def evaluate(dataloader):
    
    y_pred = torch.tensor([]) # store prediction
    y_true = torch.tensor([]) # store true label
    
    model.eval()
    with torch.no_grad():
        for label, text, offsets in dataloader:
            y_pred_batch = model(text, offsets)
            
            y_pred = torch.cat((y_pred, y_pred_batch.squeeze()))
            y_true = torch.cat((y_true, label.squeeze()))
            
    return y_pred, y_true


# In[37]:


# ======== Step 3 ==============
epochs = 300
for epoch in range(epochs):
    
    for y_train, text, offsets in train_loader:
        # zero the parameter gradients
        optimizer.zero_grad()

        # calulate output and loss 
        y_pred_train = model(text, offsets)
        loss = loss_fn(y_pred_train.squeeze(), y_train.float())

        # backprop and take a step
        loss.backward()
        optimizer.step()
    
    # evaluate on validation set
    y_pred_val, y_val = evaluate(val_loader)
    loss_val = loss_fn(y_pred_val.squeeze(), y_val.float())
    
    # note when making prediction, do add sigmoid activation
    pred_label = (torch.sigmoid(y_pred_val) > 0.5).long() # find out the class prediction
    acc = (pred_label == y_val).float().sum()/y_val.shape[0]
    
    model.train() # because when evaluating we change mode to eval mode
    
    print('Epoch {}: {:.4f} (train), {:.4f} (val), {:.4f} (val acc)'.format(epoch, loss, loss_val, acc))


# In[38]:


# prediction on test data
y_pred_test, y_true_test = evaluate(test_loader)
y_pred_test = torch.sigmoid(y_pred_test) > 0.5

print(confusion_matrix(y_true_test, y_pred_test))
print(classification_report(y_true_test, y_pred_test))

