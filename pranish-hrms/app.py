import streamlit as st
import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Nepal law Assistant")
st.title("Pranish labour Law")

with open("law_data.json", "r", encoding="utf-8") as file:
    law_data = json.load(file)

df_law = pd.DataFrame(law_data)

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-V2')

model = load_model()

def create_embeddings():
    questions = df_law['question'].tolist()
    embeddings = model.encode(questions)
    return embeddings

question_embeddings = create_embeddings()

def find_best_match(user_query):
    user_embedding = model.encode([user_query])
    similarities = cosine_similarity(user_embedding, question_embeddings)
    best_match_index = np.argmax(similarities)
    return df_law.loc[best_match_index, 'answer']

user_input = st.text_input("Ask your question:")

if st.button("Get Answer"):
    if user_input:
        answer = find_best_match(user_input)
        st.success(answer)
    else:
        st.warning("Please enter a valid question..")