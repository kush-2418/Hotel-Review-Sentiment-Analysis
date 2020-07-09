#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 23:08:56 2020

@author: kush
"""


import streamlit as st
import pickle
import numpy as np
import nltk
nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
from PIL import Image
import re,string
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english')) # 
lemma = WordNetLemmatizer()
tokenizer = WordPunctTokenizer()
twitter_handle = r'@[A-Za-z0-9_]+'                         
urls = r'http[^ ]+'
combined_handle = r'|'.join((twitter_handle, urls))  
www = r'www.[^ ]+'
punctuation = r'\W+'


def main():
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    def load_models():
        model = pickle.load(open('rfc_model.pkl', 'rb'))
        vec = pickle.load(open('tfidf_vect.pkl', 'rb'))

        return model,vec
    

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
       
    model,vec = load_models()
    img = Image.open('hotel.jpg')
    st.image(img,width=700)

    local_css("style.css")
    st.markdown("<body style='background-color:#FF0000;'></body>",unsafe_allow_html=True)
    st.markdown("<body style='background-color:#E8E8E8;'><h1 style='text-align: center; color: black;'><marquee> Hotel Review Analysis </marquee></h1></body>", unsafe_allow_html=True)

    st.markdown("<body style='background-color:#101010;'><h2 style='text-align: center; color: blue;'> Predict Reviews</h2></body>", unsafe_allow_html=True)
    st.markdown("<body style='background-color:#101010;'><h3 style='text-align: center; color: red;'> Enter the review to know whether it's a Positive or a Negative Review 👇</h3></body>", unsafe_allow_html=True)
    review = st.text_input("")

    if st.button('Predict'):
        review = clean_text(review)
        review = [review]
        review_vec = vec.transform(review).toarray()
        pred = model.predict(review_vec)

        if pred == 'negative':
            st.write("It's a Negative Review😟")
        else:
            st.write("Yehh !!! it's a Positive Review 🙂")
    
    

if __name__ == '__main__':
    main()
