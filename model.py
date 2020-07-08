#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 23:13:56 2020

@author: kush
"""


import pandas as pd
from nltk.stem import WordNetLemmatizer
import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score



df = pd.read_csv('/Users/kush/Downloads/Hotel Reviews Classification/hotel-reviews.csv')
print(df.head())

print(df['Is_Response'].value_counts())

df.drop(columns = ['User_ID', 'Browser_Used', 'Device_Used'], inplace = True)

df['Is_Response'] = df['Is_Response'].map({'happy' : 'positive', 'not happy' : 'negative'})


stop_words = set(stopwords.words('english')) # 
lemma = WordNetLemmatizer()
tokenizer = WordPunctTokenizer()
twitter_handle = r'@[A-Za-z0-9_]+'                         
urls = r'http[^ ]+'
combined_handle = r'|'.join((twitter_handle, urls))  
www = r'www.[^ ]+'
punctuation = r'\W+'

def clean_text(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()

    try:
        text = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        text = souped

    cleaned_text = re.sub(punctuation, " ",(re.sub(www, '', re.sub(combined_handle, '', text)).lower()))
    cleaned_text = ' '.join([lemma.lemmatize(word) for word in cleaned_text.split() if word not in stop_words])

    return (" ".join([word for word in tokenizer.tokenize(cleaned_text) if len(word) > 1])).strip()

df['Description'] = df['Description'].apply(lambda x: clean_text(x))

df.sample(5)

X = df.Description.values
y = df.Is_Response.values

vec = TfidfVectorizer(max_features=10000,ngram_range=(1,3))
vec.fit(X)
X_vec = vec.transform(X)


vec.get_feature_names()[:50]

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size = 0.1, random_state = 225,stratify=y)

print('X_train shape :', (X_train.shape))
print('X_test shape :', (X_test.shape))
print('y_train shape :', (y_train.shape))
print('y_test shape :', (y_test.shape))

rfc = RandomForestClassifier()
model = rfc.fit(X_train,y_train)

y_pred = model.predict(X_test) 

cm = confusion_matrix(y_pred, y_test)
print(cm)


print("Accuracy : ", accuracy_score(y_pred, y_test))
print("Precision : ", precision_score(y_pred, y_test, average = 'weighted'))
print("Recall : ", recall_score(y_pred, y_test, average = 'weighted'))


def predict_hotelreview(review):
    review= [review]
    review_vec = vec.transform(review)
    pred = model.predict(review_vec)
    return pred

import random
sample_text = random.choice(df[["Description",'Is_Response']].values.tolist())
prediction = predict_hotelreview(sample_text[0])
if prediction == 'negative':
    prediction = 'Negative'
else:
    prediction = "Positive"
print('The predicted label of the review is {}: '.format(prediction))
print('The actual label of the review is {}: '.format(sample_text[1]))

import pickle

pickle.dump(model, open('rfc_model.pkl', 'wb'))

pickle.dump(vec, open('tfidf_vect.pkl', 'wb'))