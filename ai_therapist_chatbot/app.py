##### Environment and Reproducability #####

# Core libraries
import numpy as np
import pandas as pd
import os
import warnings
import re
import gc

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# NLP & Text Processing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb

# Configuration
warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed = (RANDOM_STATE)

# Plot styling
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12,6)
plt.rcParams['font.size'] = 10

# Download nltk resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

except:
    pass

# list all available files
for dirname, _ , filenames, in os.walk('dataset'):
    for filename in filenames:
        filepath = os.path.join(dirname, filename)
        file_size_mb = os.path.getsize(filepath) /(1024 * 1024)
        print(f"{filepath} ({file_size_mb: 2f} MB)")


#######  Optimized Load Data

def load_mental_health_data():
    data_files=[]
    for dirname, _, filenames in os.walk('dataset'):
        for filename in filenames:
            data_files.append(os.path.join(dirname, filename))

    if not data_files:
        raise FileNotFoundError("No CSV found in dataset")
    
    data_path = data_files[6]
    print(f"Loading: {data_path}")

    df= pd.read_csv(
        data_path,
        dtype={'txt': 'str'},
        low_memory = False
    )

    print(f"Loaded {len(df):,} rows * {len(df.columns):,} columns successfully")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() /1024**2:.2f} MB")

    return df

df_raw = load_mental_health_data()

print("First 5 values",df_raw.head(5))
print("Column count of our dataframe", df_raw.info())


####### Data Validation #######
df = df_raw.copy()
# print(df)
text_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].str.len().mean() > 50]
potential_label_cols = [col for col in df.columns if df[col].nunique() < 20 and col.lower() in ['label', 'category', 'class', 'type', 'condition']]

if not text_cols:
    # Fallback: use first object column with reasonable length
    text_cols = [col for col in df.columns if df[col].dtype == 'object']

if not potential_label_cols:
    # Fallback: find column with few unique values
    potential_label_cols = [col for col in df.columns if 2 <= df[col].nunique() <= 20]

TEXT_COL = text_cols[0] if text_cols else df.columns[0]
LABEL_COL = potential_label_cols[0] if potential_label_cols else df.columns[-1]

print(f"Detected text columns: {TEXT_COL}")
print(f"Detected label columns: {LABEL_COL}")

# Missing Values
missing = df.isnull().sum()
missing_pct = 100 * missing / len(df)
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing_Count': missing.values,
    'Missing_Pct': missing_pct.values
}).sort_values('Missing_Pct', ascending=True)
# print(f"Missing values: {missing_df['Missing_Count'] > 0}")

print("Missing Values are:", missing_df[missing_df['Missing_Count'] > 0])

# Remove duplicate values
initial_rows = len(df)
df = df.dropna(subset=[TEXT_COL, LABEL_COL])
print(f"\n Removed {initial_rows - len(df):,} rows with missing text/label")

# Text length Distribution
df['text_length'] = df[TEXT_COL].astype(str).str.len()
print("Text length: \n", df['text_length'].describe())

# Filter out exttemely short texts
min_length = 10
df = df[df['text_length'] >= min_length]
print(f"Removed less than {min_length} characters")
print(f"New dataframe: {df}")


# Label distribution
print("Label Distribution:")
label_dist = df[LABEL_COL].value_counts()
print(label_dist)
print("Number of classes: ", df[LABEL_COL].nunique())


## Check for data leakage:
print("Data Leakage Check:")
suspicious_cols = [col for col in df.columns if 'id' in col.lower() or 'index' in col.lower()]

if suspicious_cols:
    print(f"Suspicious ID Cols found {suspicious_cols}")
else:
    print("No suspicious cols found")

print(f"Final dataset shape: {df.shape}")


####### High Signal EDA ########

# Target Distribution
fig, axes = plt.subplots(1,2, figsize=(15,5))

#Count Plot
label_counts = df[LABEL_COL].value_counts()
ax1 = axes[0]
label_counts.plot(kind='bar', ax=ax1, color='steelblue', edgecolor='black')
ax1.set_title('Target Distribution (Absolute Counts)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Mental Health Category', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.tick_params(axis='x', rotation=45)

# Add percentile annotations
total = len(df)
for i, (label, count) in enumerate(label_counts.items()):
    ax1.text(i, count, f'{100*count/total:.1f}%', ha='center', va='bottom', fontsize=10)

# Proportion Plot
ax2 = axes[1]
label_pcts = 100 * label_counts / total
colors = sns.color_palette('Set2', len(label_pcts))
ax2.pie(label_pcts, labels=label_pcts.index, autopct='%1.1f%%', colors=colors, startangle=90)
ax2.set_title('Target Distribution (Proportions)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("img/label_distribution.png", dpi=300, bbox_inches="tight")

# Class Balance Check

imbalance_ratio = label_counts.max() / label_counts.min()
print(f"\n CLass Imbalance Ratio: {imbalance_ratio:.2f}")

if imbalance_ratio > 3:
    print("Significant class imbalance detected - will use stratified sampling and weighted metrics")
else:
    print("Classes are reasonably balanced")


# Text length by Category
fig, ax = plt.subplots(figsize=(12, 6))

df.boxplot(column = 'text_length', by = LABEL_COL, ax=ax, patch_artist=True, 
           boxprops=dict(facecolor='lightblue', color='black'),
           medianprops=dict(color='red', linewidth=2))

ax.set_title('Text Length Distribution by Mental Health Category', fontsize=14, fontweight='bold')
ax.set_xlabel('Mental Health Category', fontsize=12)
ax.set_ylabel('Text Length (characters)', fontsize=12)
plt.suptitle('')  # Remove default title
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("img/text_length_dist.png", dpi=300, bbox_inches="tight")


print("\nAverage Text Length by Category:")
length_stats = df.groupby(LABEL_COL)['text_length'].agg(['mean', 'median', 'std'])
print(length_stats.round(2))

# Sample conversation from each category
print("\n Sample Conversation by category: \n")
print("=" * 100)

for category in df[LABEL_COL].unique()[:5]:
    sample = df[df[LABEL_COL] == category][TEXT_COL].iloc[0]
    print(f"category: {category}")
    print(f"Sample: {sample[:300]}..." if len(sample) > 300 else f"Sample: {sample}")
    print("-" * 100)

