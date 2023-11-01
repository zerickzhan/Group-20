#!/usr/bin/env python
# coding: utf-8

# # Group20 Assignment 

# loading the data and importing the library 

# In[46]:


import sweetviz as sv


# In[47]:


import pandas as pd 
import pytz
import numpy as np
import seaborn as sns


# In[48]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torchtext.data.utils import get_tokenizer

import matplotlib.pyplot as plt

import torch
from torch import nn 
import torch.nn.functional as F
from torch import utils

torch.manual_seed(0)
np.random.seed(0)

import warnings
warnings.filterwarnings("ignore")


# In[49]:


train=pd.read_csv('https://raw.githubusercontent.com/zerickzhan/Group-20/main/review_train.csv')
challenge=pd.read_csv('https://raw.githubusercontent.com/zerickzhan/Group-20/main/review_challenge.csv')


# ## Univariate Analysis

# we will use sweetviz for some quick and direct EDA to get started.

# In[50]:


analyze_report = sv.analyze(train)
analyze_report.show_html('report.html')


# In[51]:


analyze_reportchallenge = sv.analyze(challenge)
analyze_reportchallenge.show_html('challenge.html')


# In[52]:


train['review_timestamp']=pd.to_datetime(train['review_timestamp'])


# In[53]:


train.info()


# ### Duplicated values <br>
# from the initial analysis we can see review_id is not a unqiue key. We want to investigate further whether these are duplicates.

# In[54]:


train[train["review_id"]==570286841737318400]


# In[55]:


train[train["review_id"]==570267562623152128]


# from the above example we can see several values are duplicated with the only difference being the negative_reason_confidence, which made the review_id not an unique key.
# Therefore, we will remove the values with lower sentiment confidence.

# In[56]:


train=train.sort_values(by=['sentiment_confidence'])
train=train.drop_duplicates(subset=['review_id'], keep='last')


# In[62]:


train = train.drop_duplicates(subset='review_id', keep="first")


# Changed sentiment analysis from words into numeric values

# In[63]:


fig, ax = plt.subplots(1,1)
ax.set(title ="Histogram ")
sns.distplot(a=train["negative_reason_confidence"], bins=20)


# In[64]:


fig, ax = plt.subplots(1,1)
ax.set(title ="Histogram ")
sns.distplot(a=train["review_timestamp"], bins=20)


# In[65]:


fig, ax = plt.subplots(1,1)
ax.set(title ="Histogram ")
sns.distplot(a=train["sentiment_confidence"], bins=20)


# ### Bivariate Analysis 
# 

# In[66]:


fig, ax = plt.subplots()
ax.set(title = "Scatter ploat of Applicant Income vs Loan Amount", xlabel="Applicant Income", ylabel="Loan Amount")
plt.scatter(x='ApplicantIncome',y='LoanAmount',data=loan)
plt.show()


# Missing values

# In[68]:


train['review'] = train.apply(lambda row: row['negative_reason'] if not pd.isna(row['negative_reason']) and row['negative_reason'] != '' else row['airline_sentiment'], axis=1)


# In[69]:


pd.crosstab(train['airline_name'], train['review'])


# In[70]:


filtered_data_number = train[train['user_name'].str.contains(r'\d', regex=True, na=False)]


# In[71]:


train[train['review_city']=='Does it really matter']


# In[72]:


user_to_city_mapping = train[train['user_timezone'] != ''].groupby('user_name')['user_timezone'].first()

# Fill in the empty review_city values using the mapping
train['user_timezone'] = train.apply(
    lambda row: user_to_city_mapping.get(row['user_name'], row['user_timezone']),
    axis=1
)


# In[73]:


train[train['user_name']=="somekidnamedjon"]


# we need to identify spams and elimination of data that attribute nothing to the sentimente analysis:<br>
# we found that news and user spams are both included in the database. 
# In the mean time we also don't see much values in the timezone. As user time zone does not impact sentiment of their 
# Therefore, we decide to not include that column too. 
# 
# We would also get ride off the he thumb_up data as it contains a 94% zero value.

# creat a column that is a count of occurance of the user name  

# In[74]:


train['user_name_count'] = train.groupby('user_name')['user_name'].transform('count')


# In[76]:


user_name_count = train['user_name_count'].value_counts()

# Create a pie chart
plt.figure(figsize=(6, 6))
plt.pie(user_name_count, labels=user_name_count.index, autopct='%1.1f%%', startangle=90)
plt.title('User Name Count Distribution')
plt.show()


# I want to see the distribution for airline sentiment where only user name that contained numbers are selected.

# In[77]:


sentiment_counts=filtered_data_number['airline_sentiment'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Airline Sentiment Distribution')


# In[78]:


user_name_counts = train['user_name'].value_counts()
filtered_df = train[train['user_name'].isin(user_name_counts[user_name_counts<6].index)]


# In[79]:


count_blank = len(train[(train['user_timezone'].isna() | train['user_timezone'].eq('')) & (train['review_city'].isna() | train['review_city'].eq(''))])

print("Number of rows where both 'review_timestamp' and 'review_city' are blank:", count_blank)


# In[80]:


count_blank2 = len(train[
    (train['user_timezone'].isna() | train['user_timezone'].eq('')) &
    (train['review_city'].isna() | train['review_city'].eq('')) &
    (train['review_coordinates'].isna() | train['review_coordinates'].eq(''))
])

print("Number of rows where 'user_timezone', 'review_city', and 'review_coordinates' are blank:", count_blank)


# In[81]:


debug_selection = len(train[(train['negative_reason_confidence'].isna()) & (train['airline_sentiment'] == 'negative')])
print(debug_selection)


# as we can see, the reivew text for sentiments contain things like #Name and thanks. It would be resasonable for us to exclude those from our anlysis as

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

# Due to the frequent apperance of unformted and incorrect entries, we are unable to use review city and decide to remove that one.
# Due to the huge amount of missing data for the review coordianate. we decide to remove this column too.

# ## Feature Engineering 

# In[82]:


# preprocessing data
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder() 
target = le.fit_transform(train['airline_sentiment']) # convert target into integers
train['airline_sentiment'] = target
print(le.classes_) # this shows which index maps to which class


# In[83]:


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


# In[84]:


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


# In[85]:


# now we can build our torch dataset 
train_torch = SpamDataset(train_data)
val_torch = SpamDataset(val_data)
test_torch = SpamDataset(test_data)


# In[86]:


tokenizer = get_tokenizer('basic_english')


# In[87]:


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


# In[ ]:


# define a function that converts a document into tokens (represented by index)
def doc_tokenizer(doc):
    return torch.tensor([vocabulary[token] for token in tokenizer(doc)], dtype=torch.long)


# In[88]:


# define a function that converts a document into tokens in list instead of tensor
def doc_tokenizer2(doc):
    return [vocabulary[token] for token in tokenizer(doc)]


# In[89]:


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


# In[90]:


torch.manual_seed(0)

batchSize = 8
train_loader = utils.data.DataLoader(train_torch, batch_size=batchSize, shuffle=True, collate_fn=collate_batch)
val_loader = utils.data.DataLoader(val_torch, batch_size=batchSize, shuffle=True, collate_fn=collate_batch)
test_loader = utils.data.DataLoader(test_torch, batch_size=batchSize, shuffle=False, collate_fn=collate_batch)


# In[91]:


train_sentiment= list(train_data['airline_sentiment'])
train_review_text= []
for idx in range(len(train_data['review_text'])):
    token=doc_tokenizer2( list(train_data['review_text'])[idx])
    train_review_text.append(token)


# train_loader is tokenized values we processed 

# Model building

# In[92]:


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


# In[93]:


# ======= Step 2 ==========
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


# In[94]:


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


# In[71]:


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


# In[72]:


# prediction on test data
y_pred_test, y_true_test = evaluate(test_loader)
y_pred_test = torch.sigmoid(y_pred_test) > 0.5

print(confusion_matrix(y_true_test, y_pred_test))
print(classification_report(y_true_test, y_pred_test))


# using word embedding

# In[73]:


# ===== Build vocabulary =====
vocab2 = build_vocab_from_iterator(
    [tokenizer(s) for s in train],
    specials=["<unk>"]
)

vocab2.set_default_index(vocab2["<unk>"])


# In[74]:


# Function to convert a sentence to BoW sequence
vocab2_len = len(vocab2)
def index2onehot(idx, n = vocab2_len):
    eye = torch.eye(vocab2_len)
    return eye[:,idx]

def stobow(s):
    token_idxs = vocab2(tokenizer(s))
    return index2onehot(token_idxs), torch.Tensor(token_idxs).long()

# Convert all training sentences
bow_sequences = []
for s in train['review_text']:
    bow_sequences.append(stobow(s)) 


# In[75]:


# Model class
class RNN(nn.Module):
    def __init__(self, n_features, n_class, hidden_dim=20):
        super().__init__()
        # Model layers
        self.rnn = nn.RNN(n_features, hidden_dim)
        self.fc = nn.Linear(hidden_dim, n_class)
        
    def forward(self, x):
        rnn_o, h = self.rnn(x)
        fc_o = self.fc(rnn_o[-1, :]) 
        return fc_o

# init model
model = RNN(n_features=vocab2_len, n_class=vocab2_len)

# init loss
loss_function = nn.CrossEntropyLoss()

# init optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)


# In[205]:


max_iter = 300
seq_length = 2
loss_history = []
for epoch in range(max_iter):
    
    for i in range(len(bow_sequences)):
    
        Sent, tokens = bow_sequences[i]
        
        for j in range(Sent.shape[1]-seq_length):
            # loop over the two-grams
            X = Sent[:, j:j+seq_length] # j, j+1 words
            y = tokens[j+seq_length] # j+2 word as label
        
            optimizer.zero_grad()

            output = model(X.T) # pytorch requires input to be [Length, inputsize]
                        
            loss = loss_function(output, y)

            loss.backward()
            optimizer.step()

    # Collect loss at end of each iteration
    loss_history.append(loss.item())


# In[ ]:


plt.figure()
plt.plot(loss_history)


# Spam Classification 
# 

# In[95]:


# preprocessing data
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder() 
target = le.fit_transform(train['airline_sentiment']) # convert target into integers
train['airline_sentiment'] = target
print(le.classes_) # this shows which index maps to which class

train.head()


# simplified adaboost (week 11)

# for each row number in train_review_text add zero values to match the shape of desired shape.

# In[97]:


train_sentiment= list(train_data['airline_sentiment'])
train_review_text= []
for idx in range(len(train_data['review_text'])):
    train_token=doc_tokenizer2( list(train_data['review_text'])[idx])
    train_review_text.append(train_token)
y_train=np.array(train_sentiment)

val_sentiment= list(val_data['airline_sentiment'])
val_review_text= []
for idx in range(len(val_data['review_text'])):
    val_token=doc_tokenizer2( list(val_data['review_text'])[idx])
    val_review_text.append(val_token)
y_val=np.array(val_sentiment)

test_sentiment= list(test_data['airline_sentiment'])
test_review_text= []
for idx in range(len(test_data['review_text'])):
    test_token=doc_tokenizer2( list(test_data['review_text'])[idx])
    test_review_text.append(test_token)
y_test=np.array(test_sentiment)


# In[126]:


find_max_list(train_review_text)


# In[98]:


def find_max_list(list):
    list_len = [len(i) for i in list]
    return(max(list_len))

def zeropad(data):
    max_length = 40
    train_review_text_pad = [sublist + [0] * (max_length - len(sublist)) for sublist in data]
    zero_pad=np.array(train_review_text_pad)
    return zero_pad


# In[99]:


X_train=zeropad(train_review_text)
X_val=zeropad(val_review_text)
X_test=zeropad(test_review_text)


# ### Gradient boost

# In[ ]:


GBoost_clf= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
max_depth=1, random_state=0).fit(X_train, y_train)
clf.score(X_test, y_test)


# ### XGBoost

# In[100]:


from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb


# In[101]:


xgbst =xgb.XGBClassifier(objective='multi:softmax',
                       num_class=3,
                      eval_metric='mlogloss',
                      randomstate=42)


# In[102]:


tuning_parameters={
                   'learning_rate':[0.01,0.05,0.1],
    'max_depth':[2,3,4,5,6,7],
    'n_estimators':[800,1000,1500,2000],
    'subsample':[0.60,0.80,1.0],
    'reg_lambda':[0.0,0.3,0.5,0.8]
}


# In[103]:


xgbst_random = RandomizedSearchCV(xgbst, 
                                  tuning_parameters,
                                  n_iter=32,
                                  cv=5, 
                                 random_state=42,
                                 n_jobs=-1)
xgbst_random.fit(X_train,y_train)


# In[104]:


print('Best parameters found by randomized search:', xgbst_random.best_params_)


# In[105]:


xgbst_update=xgb.XGBClassifier(objective='multi:softmax',
                              num_class=3,
                              eval_metric='mlogloss',
                              **xgbst_random.best_params_)
xgbst.fit(X_train,y_train)


# In[106]:


xgb_y_val=xgbst.predict(X_val)


# In[111]:


xgb_y_val


# In[110]:


y_val


# ### Adaboost

# In[135]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

weak_learner = DecisionTreeClassifier(max_leaf_nodes=8)#this is a hyper parameter and needs to be tunned
n_estimators = 300

adaboost_clf = AdaBoostClassifier(
    base_estimator=weak_learner,#Use estimator instead of base_estimator for sklearn over version 1.2 
    n_estimators=n_estimators,
    algorithm="SAMME",
    random_state=42,
).fit(X_train, y_train)
#where X_train is the number valued review text, and the y_train is the sentiment(0,1,2)


# gradient boosting

# In[137]:


from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

dummy_clf = DummyClassifier()


def misclassification_error(y_true, y_pred):
    return 1 - accuracy_score(y_true, y_pred)


weak_learners_misclassification_error = misclassification_error(
    y_test, weak_learner.fit(X_train, y_train).predict(X_test)
)

dummy_classifiers_misclassification_error = misclassification_error(
    y_test, dummy_clf.fit(X_train, y_train).predict(X_test)
)

print(
    "DecisionTreeClassifier's misclassification_error: "
    f"{weak_learners_misclassification_error:.3f}"
)
print(
    "DummyClassifier's misclassification_error: "
    f"{dummy_classifiers_misclassification_error:.3f}"
)


# In[138]:



boosting_errors = pd.DataFrame(
    {
        "Number of trees": range(1, n_estimators + 1),
        "AdaBoost": [
            misclassification_error(y_test, y_pred)
            for y_pred in adaboost_clf.staged_predict(X_test)
        ],
    }
).set_index("Number of trees")
ax = boosting_errors.plot()
ax.set_ylabel("Misclassification error on test set")
ax.set_title("Convergence of AdaBoost algorithm")

plt.plot(
    [boosting_errors.index.min(), boosting_errors.index.max()],
    [weak_learners_misclassification_error, weak_learners_misclassification_error],
    color="tab:orange",
    linestyle="dashed",
)
plt.plot(
    [boosting_errors.index.min(), boosting_errors.index.max()],
    [
        dummy_classifiers_misclassification_error,
        dummy_classifiers_misclassification_error,
    ],
    color="c",
    linestyle="dotted",
)
plt.legend(["AdaBoost", "DecisionTreeClassifier", "DummyClassifier"], loc=1)
plt.show()


# In[139]:


weak_learners_info = pd.DataFrame(
    {
        "Number of trees": range(1, n_estimators + 1),
        "Errors": adaboost_clf.estimator_errors_,
        "Weights": adaboost_clf.estimator_weights_,
    }
).set_index("Number of trees")

axs = weak_learners_info.plot(
    subplots=True, layout=(1, 2), figsize=(10, 4), legend=False, color="tab:blue"
)
axs[0, 0].set_ylabel("Train error")
axs[0, 0].set_title("Weak learner's training error")
axs[0, 1].set_ylabel("Weight")
axs[0, 1].set_title("Weak learner's weight")
fig = axs[0, 0].get_figure()
fig.suptitle("Weak learner's errors and weights for the AdaBoostClassifier")
fig.tight_layout()

