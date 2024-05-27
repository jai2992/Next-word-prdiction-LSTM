import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import regex as re
import streamlit as st
import matplotlib.pyplot as plt

def file_to_sentence_list(file):
    text = file.read().decode("utf-8")
    sentences = [sentence.strip() for sentence in re.split(
        r'(?<=[.!?])\s+', text) if sentence.strip()]
    return sentences

@st.cache_data
def train_model(file_path):
    text_data = file_to_sentence_list(file_path)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_data)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in text_data:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    max_sequence_len = max([len(seq) for seq in input_sequences])
    input_sequences = np.array(pad_sequences(
        input_sequences, maxlen=max_sequence_len, padding='pre'))
    X, y = input_sequences[:, :-1], input_sequences[:, -1]

    y = tf.keras.utils.to_categorical(y, num_classes=total_words)

    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=max_sequence_len-1))
    model.add(LSTM(128))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X, y, epochs=500, verbose=1, validation_split=0.2)
    
    return model, tokenizer, max_sequence_len, history


st.title("Next Word Prediction Using LSTM")

uploaded_file = st.file_uploader("Choose a text file", type=["txt"])
if uploaded_file is not None:
    model, tokenizer, max_sequence_len, history = train_model(uploaded_file)
    st.success("Model trained successfully!")


    seed_text = st.text_input("Start typing here")
    next_words = st.slider('Select number of words to predict', min_value=1, max_value=10)

    if st.button("Predict"):
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
            predicted_probs = model.predict(token_list, verbose=0)
            predicted_index = np.argmax(predicted_probs)
            if predicted_index in tokenizer.index_word:
                predicted_word = tokenizer.index_word[predicted_index]
            else:
                predicted_word = '<unknown>'
            seed_text += " " + predicted_word
        st.success(seed_text)
        if st.button('Download model'):
            model.save('save_model.h5')
