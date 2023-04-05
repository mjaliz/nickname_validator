#!/usr/bin/env python
# coding: utf-8

# In[7]:


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from hazm import Normalizer, Lemmatizer, word_tokenize
from cleantext import clean

import pandas as pd
import numpy as np
import re
import json
import faiss
import time

# In[8]:


df_nickname = pd.read_csv('nicknames.csv')

# In[9]:


# In[10]:


df_nickname.dropna(inplace=True)


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def cleaning(text):
    text = text.strip()

    # regular cleaning
    text = clean(text,
                 fix_unicode=True,
                 to_ascii=False,
                 lower=True,
                 no_line_breaks=True,
                 no_urls=True,
                 no_emails=True,
                 no_phone_numbers=True,
                 no_numbers=False,
                 no_digits=False,
                 no_currency_symbols=True,
                 no_punct=False,
                 replace_with_url="",
                 replace_with_email="",
                 replace_with_phone_number="",
                 replace_with_number="",
                 replace_with_digit="0",
                 replace_with_currency_symbol="",
                 )

    # cleaning htmls
    text = cleanhtml(text)

    # normalizing
    normalizer = Normalizer()
    text = normalizer.normalize(text)

    # removing wierd patterns
    wierd_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u'\U00010000-\U0010ffff'
                               u"\u200d"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\u3030"
                               u"\ufe0f"
                               u"\u2069"
                               u"\u2066"
                               # u"\u200c"
                               u"\u2068"
                               u"\u2067"
                               "]+", flags=re.UNICODE)

    text = wierd_pattern.sub(r'', text)

    # removing extra spaces, hashtags
    text = re.sub("#", "", text)
    text = re.sub("\s+", " ", text)

    return text


df_nickname['nick_name'] = df_nickname['nick_name'].astype(str)
df_nickname['preprocessed_nickname'] = df_nickname['nick_name'].apply(cleaning)

df_en = pd.read_csv('clean_data.csv')
df_fa = pd.read_csv('clean_data_fa.csv')
df_sw = pd.read_csv('swear_words.csv')

df_offensive = pd.concat([df_en, df_fa, df_sw], ignore_index=True)

model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

sentences = df_offensive[df_offensive['is_offensive'] == 1]['text'].astype(str).tolist()

offensive_sentences = model.encode(sentences)

index = faiss.IndexFlatL2(offensive_sentences.shape[1])
index.add(offensive_sentences)
faiss.write_index(index, 'index_offensive_sentences')

