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
