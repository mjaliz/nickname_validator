import pandas as pd
import numpy as np
import textdistance
import re
from collections import Counter
import os

dataset_path = os.path.join('..', 'dataset')
df_sw = pd.read_csv(os.path.join(dataset_path, 'swear_words.csv'))

swear_words = df_sw['text'].astype(str).tolist()


def tokenizer(word):
    re_pattern = r'\b\w+\b'
    tokens = re.findall(re_pattern, word)
    return tokens


class SpellChecker:
    def __init__(self):
        self.__V = set(swear_words)
        self.__word_freq = {}
        self.__word_freq = Counter(swear_words)
        self.__probs = {}
        total = sum(self.__word_freq.values())
        for k in self.__word_freq.keys():
            self.__probs[k] = self.__word_freq[k] / total

    def spell_check(self, input_word):
        input_word = input_word.lower()
        if input_word in self.__V:
            return input_word
        else:
            sim = [1 - (textdistance.Jaccard(qval=2).distance(v, input_word)) for v in self.__word_freq.keys()]
            df = pd.DataFrame.from_dict(self.__probs, orient='index').reset_index()
            df = df.rename(columns={'index': 'Word', 0: 'Prob'})
            df['Similarity'] = sim
            output = df.sort_values(['Similarity', 'Prob'], ascending=False).head()
            most_similar = output.iloc[0]
            if most_similar['Similarity'] > 0.5:
                return most_similar['Word']
            return input_word


class RuleBaseChecker:
    def __init__(self):
        self.__swear_words = swear_words
        self.__spell_checker = SpellChecker()

    def find_swear(self, input_word):
        word = self.__spell_checker.spell_check(input_word)

        email_pattern = r'[\w.+-]+@[\w-]+\.[\w.-]+'
        email_found = re.findall(email_pattern, word)
        if email_found:
            print('email address')
            return 'email address'

        tokens = tokenizer(word)
        for swear_word in self.__swear_words:
            if swear_word in tokens:
                print(swear_word)
                return swear_word

        for token in tokens:
            mobile_pattern = r'^(\+98|0)?9\d{9}$'
            mobile_found = re.findall(mobile_pattern, token)
            if mobile_found:
                print('phone number')
                return 'phone number'

        for swear_word in self.__swear_words:
            match = re.search(swear_word, word)
            if match:
                print(match)
                return match.group(0)

        return 0
