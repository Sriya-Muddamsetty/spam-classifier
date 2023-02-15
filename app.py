import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import sklearn
nltk.download('punkt')

ps=PorterStemmer()
def transform_text(text):
    text = text.lower()  # lower
    text = nltk.word_tokenize(text)  # tokenization

    y = []

    # Removing special characters

    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    # removing stopwords and punctuation

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    # stemming

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model= pickle.load(open('model.pkl','rb'))
st.title("SMS Spam Classifier")
input_sms=st.text_input("Enter the message below")
if st.button('Classify'):

    transformed_sms=transform_text(input_sms)
    vector_input=tfidf.transform([transformed_sms])
    result=model.predict(vector_input)[0]
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")

