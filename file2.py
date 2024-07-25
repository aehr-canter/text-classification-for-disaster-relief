#different models to try out
#functions for fitting logistic regression, KNN, decision trees, etc.
#evaluation metrics like accuracy, precision recall

from file1 import tweets_haiti, tweets_sandy
import nltk
import pandas as pd
import collections
import numpy as np
import csv
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy

nlp = spacy.load('en_core_web_sm')



# initialize a list to store labels, with "NaN" as the default value
#labelsResults = ["NaN"] * len(tweets_haiti)

keywords = {
    "Food": ["food", "hunger", "meal", "groceries", "nutrition", "eat", "hungry", "meals"],
    "Water": ["water", "thirst", "thirsty", "hydration", "drink"],
    "Energy": ["energy", "power", "electricity", "fuel", "gas", "battery", "heat"],
    "Medical": ["medical", "health", "medicine", "doctor", "hospital", "clinic", "treatment", "virus"]
}



#preprocessing function using spaCy
def preprocess_text_spacy(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'\W', ' ', tweet)
    tweet = re.sub(r'\d+','', tweet)
    doc = nlp(tweet)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

#preprocess haiti dataset
tweets_haiti_preprocessed = [preprocess_text_spacy(tweet) for tweet in tweets_haiti]

#label assignment function
def assign_label(words, keywords):
    for label, kw_list in keywords.items():
        if any(word in words for word in kw_list):
            return label
    return "NaN"

#assign labels to haiti dataset
labelsResults_haiti = [assign_label(words, keywords) for words in tweets_haiti_preprocessed]
print('Labels for Haiti Dataset: ', labelsResults_haiti)

#Preprocess sandy dataset
tweets_sandy_preprocessed = [preprocess_text_spacy(tweet) for tweet in tweets_sandy]

#assign labels to sandy dataset
labelsResults_sandy = [assign_label(words, keywords) for words in tweets_sandy_preprocessed]
print('\nLabels for Sandy Dataset: ', labelsResults_sandy)
