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

all_tokens = df['tokens'].explode()

# Skill Embeddings
skills = (
    all_tokens
    .value_counts()
    .head(500)
    .index
    .tolist()
)


skill_embeddings = model.encode(skills)

# Group Skills together
kmeans = KMeans(
    n_clusters=10,
    random_state=42
)

labels = kmeans.fit_predict(skill_embeddings)

for skill, cluster in zip(skills, labels):
    print(skill, cluster)

# Identify domain-specific competencies
clusters = {}

for skill, label in zip(skills, labels):
    clusters.setdefault(int(label), []).append(skill)

print(clusters)

#Compare all resumes

resume_embeddings = model.encode(
    df['Resume'].tolist(),
    show_progress_bar= True
)

similarity_matrix = cosine_similarity(
    resume_embeddings
)
print(similarity_matrix.shape)

# Find most similar resumes 
resume_index = 0

scores = similarity_matrix[resume_index]

similar_indices = np.argsort(scores)[::-1]

# Display top matches
for idx in similar_indices[1:6]:
    print(
        f"Resume ID: {df.iloc[idx]['ID']}"
    )
    print(
        f"Category: {df.iloc[idx]['Category']}"
    )
    print(
        f"Similarity: {scores[idx]:.3f}"
    )
    print("-" * 40)
