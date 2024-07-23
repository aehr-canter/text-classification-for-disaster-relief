import pandas as pd
import collections

#Load datasets
train_haiti = pd.read_csv('haiti_train.csv')
train_sandy = pd.read_csv('sandy_train.csv')

#countWordFreq
texts = train_haiti['Text'].str.cat(sep=' ')
words = texts.split()
word_counts = collections.Counter(words)
top_words = word_counts.most_common(30)
print(top_words)
