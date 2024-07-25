# #data analysis + error analysis
# #what are the frequent words/terms in the dataset?
# #what sorts of errors do the models make
# #confusion matrices/other visualizations


# import pandas as pd
# from file2 import assign_label
# from file2 import labelsResults
# from file2 import assign_label, preprocess_text_spacy
# import spacy
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# #Load datasets
# train_haiti = pd.read_csv('haiti_train.csv')
# train_sandy = pd.read_csv('sandy_train.csv')
# # convert the 'Text' column into lists of tweets
# tweets_haiti = train_haiti['Text'].tolist()
# tweets_sandy = train_sandy['Text'].tolist() 

# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(tweets_haiti)
# y = labelsResults

# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=42)

# classifier = LogisticRegression()

# classifier.fit(X_train, y_train)

# y_pred = classifier.predict(train_haiti['Label'])

# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.2f}')
# print('\nClassification Report:')
# print(classification_report(y_test, y_pred))


