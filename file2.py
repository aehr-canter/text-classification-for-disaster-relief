#different models to try out
#functions for fitting logistic regression, KNN, decision trees, etc.
#evaluation metrics like accuracy, precision recall

from file1 import tweets_haiti
# initialize a list to store labels, with "NaN" as the default value
labelsResults = ["NaN"] * len(tweets_haiti)



def assign_label():
    count = 0
    # Iterate over each tweet in the Haiti dataset
    for tweet in tweets_haiti:
        # Convert tweet to lowercase
        tweet.lower()
        # Split the tweet into individual words
        words = tweet.split()
        
        # Assign labels based on keywords found in the tweet
        if "food" in words:
            labelsResults[count] = "Food"
        elif "water" in words:
            labelsResults[count] = "Water"
        elif "energy" in words:
            labelsResults[count] = "Energy"
        elif "medical" in words:
            labelsResults[count] = "Medical"

        # Increment the count to move to the next label
        count+=1
        # Print the final list of labels for the Haiti tweets
        print(labelsResults)
        #checking if it worked for each tweet (total 1601)
        print(len(labelsResults))

assign_label()

#countWordFreq
# words = texts.split()
# word_counts = collections.Counter(words)
# top_words = word_counts.most_common(30)
# print(top_words)