#different models to try out
#functions for fitting logistic regression, KNN, decision trees, etc.
#evaluation metrics like accuracy, precision recall

from file1 import tweets_haiti
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

def preprocess_text_spacy(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'\W', ' ', tweet)
    tweet = re.sub(r'\d+','', tweet)
    doc = nlp(tweet)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

tweets_haiti_preprocessed = [preprocess_text_spacy(tweet) for tweet in tweets_haiti]

def assign_label(words, keywords):
    for label, kw_list in keywords.items():
        if any(word in words for word in kw_list):
            return label
    return "NaN"

labelsResults = [assign_label(words, keywords) for words in tweets_haiti_preprocessed]
print(labelsResults)


#     count = 0
#     # Iterate over each tweet in the Haiti dataset
#     for tweet in tweets_haiti:
#         # Convert tweet to lowercase
#         tweet.lower()
#         # Split the tweet into individual words
#         words = tweet.split()
        
#         # Assign labels based on keywords found in the tweet
#         if "food" in words:
#             labelsResults[count] = "Food"
#         elif "water" in words:
#             labelsResults[count] = "Water"
#         elif "energy" in words:
#             labelsResults[count] = "Energy"
#         elif "medical" in words:
#             labelsResults[count] = "Medical"

#         # Increment the count to move to the next label
#         count+=1
#         # Print the final list of labels for the Haiti tweets
#         print(labelsResults)
#         #checking if it worked for each tweet (total 1601)
#         print(len(labelsResults))

# #assign_label()
