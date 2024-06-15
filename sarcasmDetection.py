#Let's start with importing the needed libraries

import pandas as pd 
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer 
# The CountVectorizer class is used to convert a collection of text documents to a matrix of token counts:
# 1) it tokenizes the string (splits the string into individual words) and gives an integer ID to each token. 
# 2) It counts the occurrence of each of those tokens.

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

#Importing the dataset

data = pd.read_json("Sarcasm-Data.json")
print(data.head())

#The "is_sarcastic" column in the dataset contains the labels that we have to predict for the task of sarcasm detection. It contains
#binary values 0 and 1, where 0 represents non-sarcastic and 1 represents sarcastic. For simplicity, I will transform the values of this column as "sarcastic" and "not sarcastic" instead of 1 and 0

data["is_sarcastic"] = data["is_sarcastic"].map({0: "Not Sarcasm", 1: "Sarcasm"})
print(data.head())

#Now I will prepare the data for training a machine learning model. This dataset has three columns, out of which we only need the "headline" column as a feature and the "is_sarcastic" column
# as a label. Let's select these columns and split the data into 20% test set and 80% training set.

data = data[["headline", "is_sarcastic"]]
x = np.array(data["headline"])
y = np.array(data["is_sarcastic"])

cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#The random_state parameter is used to control the randomness of the algorith. It's used for reproducibility of the results. 
#In this case it controls the shuffling applied to the data before applying the solit.

#I will use the Bernoulli Naive Bayes algorithm to train a model for the task of sarcasm detection 
#It is a supervised machine learning algorithm. It uses the Bayes Theorem to predict the posterior probability of any event based on the events that have already
#occurred. Naive Bayes is used to perform classification and assumes that all the events are independent.
#The Bayes theorem is used to calculate the conditional probability of an event, given that another event has already occurred. 

#Bernoulli distribution is used for discrete probability calculation. It either calculates success or failure. 

#Bernoulli Naive Bayes is a subcategory of the Naive Bayes Algorithm. It is used for the classification of binary features such as "Yes" or "No", "True" or "False", "Success" or "Failure", etc. This algorithm
#is typically used for spam detection, text classification, Sentiment Analysis.

model = BernoulliNB()
model.fit(X_train, y_train)
print(model.score(X_test, y_test)) #Output: 0.8448146761512542

#The accuracy of the model is approximately 84.48%. This means that the model can predict whether a headline is sarcastic or not with an accuracy of 84.48%.   
#The score method returns the accuracy of the model on the given test data and labels. It's a measure of how well the model performs on unseen data.In this case the model is quite good at
#predicting the target variable based on the features in the test dataset.

#Let's use a sarcastic text as input to test whether our machine learning model detects sarcasm or not:

user = input("Enter a text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)