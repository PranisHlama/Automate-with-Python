import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# PAGE CONFIG
st.set_page_config(page_title="Electricity Authority Chatbot")
st.title("Electricity Authority Chatbot")

# DATAA

qa_data = [
    {
        'question': 'How do I report a power outage',
        'answer': 'You can report a power outage by calling our 24/7 hotline at 1-800-POWER-OUT (1-800-769-3768)'
    },
    {
        'question': 'Where can I check for current power outage?',
        'answer': 'Information on renewable energy options, including solar panel installation guidelines, net me'
    }
]

df_qa = pd.DataFrame(qa_data)

# LOAD MODEL

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-V2')

model = load_model()

# CREATE EMBEDDINGS (cached)

def create_embeddings():
    questions = df_qa['question'].tolist()
    embeddings = model.encode(questions)
    return embeddings

question_embeddings = create_embeddings()

# FIND BEST MATCH FUNCTION

def find_best_match(user_query):
    user_embedding = model.encode([user_query])
    similarities = cosine_similarity(user_embedding, question_embeddings)
    best_match_index = np.argmax(similarities)
    return df_qa.loc[best_match_index, 'answer']

# CHAT UI
user_input = st.text_input("Ask your question:")

if st.button("Get Answer"):
    if user_input:
        answer = find_best_match(user_input)
        st.success(answer)
    else:
        st.warning("Please enter a question.")
    
