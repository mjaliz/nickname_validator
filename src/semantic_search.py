from sentence_transformers import SentenceTransformer
from hazm import Normalizer
from cleantext import clean

import pandas as pd
import re
import faiss
import os

import spacy

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_lg")

dataset_path = os.path.join('..', 'dataset')

df_en = pd.read_csv(os.path.join(dataset_path, 'clean_data.csv'))
df_fa = pd.read_csv(os.path.join(dataset_path, 'clean_data_fa.csv'))
df_sw = pd.read_csv(os.path.join(dataset_path, 'swear_words.csv'))


def cleanhtml(raw_html):
    cleaner = re.compile('<.*?>')
    cleantext = re.sub(cleaner, '', raw_html)
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
                 no_punct=True,
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


df_offensive = pd.concat([df_en, df_fa, df_sw], ignore_index=True)
df_offensive['text'] = df_offensive['text'].astype(str)

model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

sentences = df_offensive[df_offensive['is_offensive'] == 1]['text'].astype(str).tolist()

if not os.path.isfile('index_offensive_sentences'):
    print('Encoding sentences ...')
    offensive_sentences = model.encode(sentences)

    # Create an index using FAISS
    index_sentences = faiss.IndexFlatL2(offensive_sentences.shape[1])
    index_sentences.add(offensive_sentences)
    faiss.write_index(index_sentences, 'index_offensive_sentences')

index_sentences = faiss.read_index('index_offensive_sentences')


def search_sentence(query):
    doc = nlp(query)
    for token in doc:
        query_vector = model.encode([token.text])
        k = 5
        top_k = index_sentences.search(query_vector, k)
        min_dist = top_k[0].min()
        if min_dist < 0.05:
            return 1
    return 0
