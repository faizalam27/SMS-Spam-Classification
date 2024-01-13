import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import numpy as np
pip install nltk
pip install streamlit
pip install pickle


ps = PorterStemmer()

cv = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('IS HE/SHE A SPAM PERSON?')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]  # here we are cloning
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]  # here we are cloning
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

input_sms = st.text_area('Enter the message')

if st.button('Predict'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = cv.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("CAREFUL: HE/SHE IS SPAM")
    else:
        st.header("CONGRATS: HE/SHE IS GOOD")
