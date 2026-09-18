import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------
# LOAD MODEL + TOKENIZER
# ---------------------------
model = load_model("nextword_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Rebuild index-to-word dictionary
index_to_word = {index: word for word, index in tokenizer.word_index.items()}

# ---------------------------
# CONFIG
# ---------------------------
max_length = 50  # ⚠️ set to the same max_length used during training

def predict_next_word(seed_text, next_words=3):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_length-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)[0]
        next_word = index_to_word.get(predicted, "")
        seed_text += " " + next_word
    return seed_text

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("Next Word Prediction")

seed_text = st.text_input("Enter seed text:", "data science")
next_words = st.slider("Number of words to predict:", 1, 10, 3)

if st.button("Predict"):
    result = predict_next_word(seed_text, next_words)
    st.success(result)
