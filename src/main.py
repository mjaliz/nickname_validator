from rule_base_search import RuleBaseChecker
import pandas as pd
import os

dataset_path = os.path.join('..', 'dataset')
df = pd.read_csv(os.path.join(dataset_path, 'nicknames.csv'))

swf = RuleBaseChecker()

df['nick_name'] = df['nick_name'].astype(str)
df['is_offensive'] = df['nick_name'].apply(swf.find_swear)
df.to_csv('swears.csv', index=None)