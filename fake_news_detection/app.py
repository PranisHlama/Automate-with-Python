import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer

import re
from collections import Counter

df1 = pd.read_csv('news_dataset/Fake.csv')
df2 = pd.read_csv('news_dataset/True.csv')


df1['label'] = 'real'
df2['label'] = 'fake'

df = pd.concat([df1, df2], ignore_index=False)

# print(df1.isnull().sum())
# df1.dropna()

# print("Fake:\n", df1.head(10))
# print(df1.shape)
# df1.info()
# print(df1.sample(5)) # Gives random Rows
# print(df1.dtypes)

print(df.isnull().sum())

print(df.shape)

df.info()

print(df.head(20))

print(df.sample(10))

df['article_length'] = df['text'].apply(lambda x: len(x.split()))

average_length = df['article_length'].mean()

print("Average length of article: ", average_length)


label_counts = df['label'].value_counts(normalize=True) * 100

print(label_counts)

label_counts.plot(kind='bar')

plt.title("Label Distribution")
plt.xlabel("Label")
plt.ylabel("Number of Articles")
plt.savefig("img/label_distribution.png", dpi=300, bbox_inches="tight")


stop_words = set(ENGLISH_STOP_WORDS)

def clean_text_remove_stopwords(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return words

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    return words

def get_common_words(df, text_column='text', top_n=20):
    all_words = []

    for article in df['text']:
        all_words.extend(clean_text_remove_stopwords(article))

    word_counts = Counter(all_words)

    common_words_df = pd.DataFrame(
        word_counts.most_common(top_n), columns=['word', 'count']
    )

    return common_words_df

# Fake common words
fake_common_words = get_common_words(
    df[df['label'] == 'fake'], 
    text_column='text', 
    top_n=20)
# fake_common_words.plot(kind="bar", x="word", y="count", legend=False)

plt.figure(figsize=(10, 6))

plt.barh(fake_common_words["word"], fake_common_words["count"])

plt.title("Top 20 Common Words in Fake News Articles")
plt.xlabel("Frequency")
plt.ylabel("Word")

plt.gca().invert_yaxis()  # highest count appears at the top
plt.tight_layout()

plt.savefig("img/fake_common_words.png", dpi=300, bbox_inches="tight")

# Real Common Words
real_common_words = get_common_words(
    df[df['label'] == 'real'],
    text_column='text',
    top_n=20)

plt.figure(figsize=(10, 6))

plt.barh(real_common_words["word"], real_common_words["count"])

plt.title("Top 20 Common Words in Real News Articles")
plt.xlabel("Frequency")
plt.ylabel("Word")

plt.gca().invert_yaxis()  # highest count appears at the top
plt.tight_layout()

plt.savefig("img/real_common_words.png", dpi=300, bbox_inches="tight")

def get_common_phrases(text_data, ngram_range, top_n):
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=ngram_range,
        max_features=top_n
    )

    X = vectorizer.fit_transform(text_data.astype(str))

    phrase_counts = X.sum(axis=0)
    phrases = vectorizer.get_feature_names_out()

    phrase_freq = []

    for phrase, count in zip(phrases, phrase_counts.tolist()[0]):
        phrase_freq.append((phrase, count))

    phrase_freq = sorted(phrase_freq, key=lambda x: x[1], reverse=True)

    return pd.DataFrame(phrase_freq, columns=["phrase", "count"])

# Identify common 2-phrases using CountVectorizer
bigram_df = get_common_phrases(df["text"], ngram_range=(2, 2), top_n=20)

print("Bigram range: \n", bigram_df)

# Identify trigram 3 phase words using CountVectorizer
trigram_df = get_common_phrases(df["text"], ngram_range=(3, 3), top_n=20)

print("Trigram Range: \n", trigram_df)

fake_phrases = get_common_phrases(
    df[df['label'] == 'fake']['text'],
    ngram_range=(2, 2),
    top_n=20
)
real_phrases = get_common_phrases(
    df[df['label'] == 'real']['text'],
    ngram_range=(2, 2),
    top_n=20
)

print("Fake News Common Phrases")
print("Fake Phrases: \n",fake_phrases)

print("Real News Common Phrases")
print("Real Phrases: \n", real_phrases)