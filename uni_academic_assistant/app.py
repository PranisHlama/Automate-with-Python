import numpy as np
import pandas as pd
import streamlit as st
import json
import faiss
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="University Chat Assistant")
st.title("Pranish Uni Assistant")

with open("uni_data.json", "r", encoding="utf-8") as file:
    uni_data = json.load(file)

print("before logic: \n",uni_data)

# Load Data
rows = []

for key, value in uni_data.items():
    if isinstance(value, dict):
        if "question" in value and "answer" in value:
            rows.append({
                "section": key,
                "question": value["question"],
                "answer": value["answer"]
            })
        
        else:
            for subkey, subvalue in uni_data.items():
                if isinstance(subvalue, dict) and "question" in subvalue:
                    rows.append({
                        "section": subkey,
                        "question": subvalue["question"],
                        "answer": subvalue["answer"]
                    }) 
    elif isinstance(value, list):
        for item in value:
            rows.append({
                "section": key,
                "question": item["question"],
                "answer": item["answer"]
            })
        
print("After logic:\n", rows)

df_uni = pd.DataFrame(rows)

# Load Embedding Model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-distilroberta-v1')

model = load_model()


# Create Embedding
def create_embeddings():
    questions = df_uni["question"].tolist()
    embeddings = model.encode(questions).astype("float32")
    return embeddings


# Build FAISS Index
question_embeddings = create_embeddings()
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(question_embeddings).astype("float32"))


# RAG Function
def rag_answer(user_query):
    user_embedding = model.encode([user_query]).astype("float32")
    # Retrieve top 3 matches
    D, I = index.search(user_embedding, 3)

    # Build Context
    contexts = []
    for i in I[0]:
        contexts.append(df_uni.loc[i, "answer"])
    
    context_text = "\n".join(contexts)

    prompt = f"""
        You are a university support assistant who helps students with queries.
        Rules:
        - Only use the given context
        - If answer is not in context, say "I don't have enough information in university data"

        Context: 
        {context_text}

        Question: 
        {user_query}
    """

    # Generate Responses using LLM
    response = ollama.chat(
        model = "mistral",
        messages=[
            {"role": "system", "content": "You are a helpful university Assistane."},
            {"role": "user", "content": prompt}
        ]
    )

    
    return response["message"]["content"], contexts
    # best_match_index = I[0][0]
    # return df_uni.loc[best_match_index, "answer"]

# Streamlit UI
if "chat" not in st.session_state:
    st.session_state.chat = []

with st.form("question_form"):
    user_input = st.text_input("Ask your question:")
    submitted = st.form_submit_button("Get Answer")

# user_input = st.text_input("Ask your question:")

if submitted:
    if user_input:
        answer, sources = rag_answer(user_input)
        st.session_state.chat.append(("user", user_input))
        st.session_state.chat.append(("bot", answer))
        
        # st.subheader("Retrieved Context")
        for s in sources:
            st.info(s)

for role, msg in st.session_state.chat:
    if role =="user":
        st.markdown(f"**User:** {msg}")
    else:
        st.markdown(f"**Assistant:** {msg}")