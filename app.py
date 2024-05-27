import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import numpy as np
import pickle
import regex as re

st.title("Next Word Prediction Using LSTM")


def file_to_sentence_list(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    sentences = [sentence.strip() for sentence in re.split(
        r'(?<=[.!?])\s+', text) if sentence.strip()]

    return sentences

file_path = 'data.txt'
text_data = file_to_sentence_list(file_path)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
total_words = len(tokenizer.word_index) + 1


model = load_model('model.h5')
max_sequence_len = 81

seed_text = st.text_input("Start typing here")
next_words = st.slider('Select number of words to predict', min_value=1, max_value=10)

if st.button("Predict"):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs)
        predicted_word = tokenizer.index_word.get(predicted_index, '<unknown>')
        seed_text += " " + predicted_word
    st.success(seed_text)
