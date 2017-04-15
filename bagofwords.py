#!/usr/bin/env python
#Bag of words with vocabulary size = 5000 most frequent words

import os,sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np

if __name__ == '__main__':
    train = pd.read_csv('1-restaurant-train.csv', sep='\t', header=0)
    test = pd.read_csv('1-restaurant-test.csv', sep='\t', header=0 )

    print ('The first review is:')
    print (train["review"][0])
    clean_train_reviews = []

    # Loop over each review; create an index i that goes from 0 to the length
    # of the restaurant review list

    print ("Cleaning and parsing the training set movie reviews...\n")
    for i in range( 0, len(train["review"])):
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))

    print ("Creating the bag of words...\n")


    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 10000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    train_data_features = vectorizer.fit_transform(clean_train_reviews)

    # Numpy arrays are easy to work with, so convert the result to an
    # array
    np.asarray(train_data_features)
    # Take a look at the words in the vocabulary
    #vocab = vectorizer.get_feature_names()
    #print vocab
    # ******* Train a random forest using the bag of words
    #
    print ("Training the random forest (this may take a while)...")


    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators = 200)

    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    forest = forest.fit( train_data_features, train["sentiment"] )



    # Create an empty list and append the clean reviews one by one
    clean_test_reviews = []

    print ("Cleaning and parsing the test set movie reviews...\n")
    for i in range(0,len(test["review"])):
        clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    np.asarray(test_data_features)

    # Use the random forest to make sentiment label predictions
    print ("Predicting test labels...\n")
    result = forest.predict(test_data_features)

    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    #output = pd.DataFrame( data={"Id":test["id"], "Solution":result} )
    output = pd.DataFrame( data={"sentiment":result} )

    # Use pandas to write the comma-separated output file
    output.to_csv('Bag_of_Words_model.csv', index=False, quoting=3)
    print ("Wrote results to Bag_of_Words_model.csv")

#Error: 225 685 0.328467153285

#Error Multiple: 271.25 685 0.39598540146

