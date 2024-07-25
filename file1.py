#feature engineering, data preprocessing
#TFIDF features, CountVectorizer features, etc
#removing missing values from dataset, prepare dataset to be fed into model

import pandas as pd
import collections
import numpy as np
import csv
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_extraction.text import CountVectorizer

#Load datasets
train_haiti = pd.read_csv('haiti_train.csv')
train_sandy = pd.read_csv('sandy_train.csv')
# convert the 'Text' column into lists of tweets
tweets_haiti = train_haiti['Text'].tolist()
tweets_sandy = train_sandy['Text'].tolist()
