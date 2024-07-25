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
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, accuracy_score

nlp = spacy.load('en_core_web_sm')



# initialize a list to store labels, with "NaN" as the default value
#labelsResults = ["NaN"] * len(tweets_haiti)
#Bag of Words Model
keywords = {
    "Food": ["food", "hunger", "meal", "groceries", "nutrition", "eat", "hungry", "meals", "feed", "dish"],
    "Water": ["water", "thirst", "thirsty", "hydration", "drink"],
    "Energy": ["energy", "power", "electricity", "fuel", "gas", "battery", "heat", "lights", "light", "batteries"],
    "Medical": ["medical", "health", "medicine", "doctor", "hospital", "clinic", "treatment", "virus", "painkiller", "painkillers", "generator", ]
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



#Naive Bayes Classifier
def naive_bayes_classifier(train_texts, train_labels, test_texts, test_labels):
    # Vectorize the text data
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, train_labels)
    predicted_labels = nb_classifier.predict(X_test)
    accuracy = accuracy_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels, average='weighted')
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")

train_texts = tweets_haiti_preprocessed
train_labels = labelsResults_haiti
test_texts = tweets_sandy_preprocessed
test_labels = labelsResults_sandy

naive_bayes_classifier(train_texts, train_labels, test_texts, test_labels)