import streamlit as st
import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Nepal law Assistant")
st.title("Bikash labour Law")

with open("law_data.json", "r", encoding="utf-8") as file:
    law_data = json.load(file)

# Convert nested JSON into rows
rows = []

for key, value in law_data.items():

    # Normal question-answer
    if isinstance(value, dict):

        if "question" in value and "answer" in value:
            rows.append({
                "section": key,
                "question": value["question"],
                "answer": value["answer"]
            })

        else:
            # Nested dictionaries
            for subkey, subvalue in value.items():

                if isinstance(subvalue, dict) and "question" in subvalue:
                    rows.append({
                        "section": subkey,
                        "question": subvalue["question"],
                        "answer": subvalue["answer"]
                    })

    # Lists like regulatory bodies
    elif isinstance(value, list):
        for item in value:
            rows.append({
                "section": key,
                "question": item["question"],
                "answer": item["answer"]
            })
print(rows)
# Create dataframe
df_law = pd.DataFrame(rows)


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