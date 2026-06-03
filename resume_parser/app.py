import pandas as pd
import numpy as np
import re

import nltk
from nltk.corpus import stopwords
import spacy

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])

df = pd.read_csv('Resume.csv')

# Step 1: Remove junk columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Step 2: Kepp required columns
df = df[['ID', 'Category', 'Resume_str']]

# Step 3: Rename properly
df.rename(columns={'Resume_str': 'Resume'}, inplace=True)

# STEP 4: handle missing values
df['Resume'] = df['Resume'].fillna("")
# df.drop(columns=['Resume_str', 'Resume_html'], inplace=True)

# Step 5: Basic Cleaning
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

df['Resume'] = df['Resume'].apply(clean_text)

# Step 6 Lemmatization + stopwords
def lemmatize_batch(texts):
    docs = nlp.pipe(texts, batch_size=100)
    return [" ".join(
        token.lemma_ for token in doc
        if token.lemma_ not in stop_words and token.lemma_.strip() != ""
        ) 
        for doc in docs
    ]

df['Resume'] = lemmatize_batch(df["Resume"].fillna("").tolist())

# step 7: tokenization
df['tokens'] = df['Resume'].apply(lambda x: x.split())

print(df.head(10))

# Model Development
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print(df['tokens'])

# Skill Embeddings
skills = [
    "strategic planning",
    "organizational development",
    "SAP",
    "IBM",
    "professional",
    "business analyst",
    "engineer",
    "leadership",
    "cybersecurity",
    "ui/ux",
    "python",
    "django",
    "react",
    "machine learning",
    "payroll"
]

embeddings = model.encode(skills)

# Group Skills together
KMeans = KMeans(
    n_clusters=5,
    random_state=42
)

labels = KMeans.fit_predict(embeddings)

for skill, cluster in zip(skills, labels):
    print(skill, cluster)

# Identify domain-specific competencies
clusters = {}

for skill, label in zip(skills, labels):
    clusters.setdefault(int(label), []).append(skill)

print(clusters)

resume_embedding_1 = model.encode(df['Resume'].iloc[0])
resume_embedding_2 = model.encode(df['Resume'].iloc[1])

similarity = cosine_similarity(
    [resume_embedding_1],
    [resume_embedding_2]
)

print(similarity)