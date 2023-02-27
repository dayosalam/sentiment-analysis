#!/usr/bin/env python
# coding: utf-8

# In[8]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re


# In[21]:


#Installize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


# In[23]:


#Encode and  calculate sentiment
tokens = tokenizer.encode('I hate this, absolutely the worst', return_tensors='pt')


# In[11]:


tokens


# In[24]:


result = model(tokens)
result


# In[25]:


int(torch.argmax(result.logits))+1


# In[28]:


#Scrapping a yelp comment section
r = requests.get('https://www.yelp.com/biz/lazy-dog-restaurant-and-bar-concord-3?hrid=sws7VMCWG9BN7z1h9jfS8g')
soup = BeautifulSoup(r.text,'html.parser')
regex = re.compile('.*comment.*')
results = soup.find_all('p', {'class': regex})
reviews = [result.text for result in results]


# In[29]:


reviews


# In[30]:


import pandas as pd
import numpy as np


# In[31]:


df = pd.DataFrame(np.array(reviews), columns=['reviews'])
df.head()


# In[35]:


def sentiment_analysis(review):
    
    tokens = tokenizer.encode(review, return_tensors='pt')
    result= model(tokens)
    return int(torch.argmax(result.logits))+1


# In[36]:


df['sentiment']=df['reviews'].apply(lambda x: sentiment_analysis(x[:512]))


# In[37]:


df


# In[ ]:




