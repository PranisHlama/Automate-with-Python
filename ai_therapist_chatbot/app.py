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

# Logistic Regression(Baseline)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Advanced Model: Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Advanced Model: XGBoost
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

##### Feature Engineering #####

# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()

    text = re.sub(r'http\S+|www\S+', '', text)

    text = re.sub(r'\S+@\S+', '', text)

    text = re.sub(r'\s+', '', text).strip()

    return text

print("Cleaning text data")
df['text_clean'] = df[TEXT_COL].apply(clean_text)

print("\n Extracting linguistic features...")

def extract_linguistic_features(text):
    features = {}

    features['word_count'] = len(text.split())
    features['char_count'] = len(text)
    features['avg_word_length'] = features['char_count'] / max(features['word_count'], 1)

    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['period_count'] = text.count('.')

    upper_chars = sum(1 for c in text if c.isupper())
    features['uppercase_ratio'] = upper_chars / max(len(text), 1)

    # Sentiment analysis using TextBlob
    try:
        blob = TextBlob(text)
        features['sentiment_polarity'] = blob.sentiment.polarity
        features['sentiment_subjectivity'] = blob.sentiment.subjectivity
    except:
        features['sentiment_polarity'] = 0.0
        features['sentiment_subjectivity'] = 0.5

     # Crisis-related keywords (domain knowledge)
    crisis_keywords = ['suicide', 'kill', 'die', 'death', 'hurt', 'harm', 'end it', 'hopeless', 'worthless']
    features['crisis_keyword_count'] = sum(keyword in text.lower() for keyword in crisis_keywords)

    # Anxiety related keywords
    anxiety_keywords = ['anxiety', 'panic', 'worry', 'nervous', 'stress', 'fear', 'scared']
    features['anxiety_keyword_count'] = sum(keyword in text.lower() for keyword in anxiety_keywords)

    # Depression Related keywords
    depression_keywords = ['depress', 'sad', 'empty', 'numb', 'tired', 'exhaust', 'hopeless']
    features['depression_keyword_count'] = sum(keyword in text.lower() for keyword in depression_keywords)

    return features

# Extract features for all texts

linguistic_features = df['text_clean'].apply(extract_linguistic_features)
df_linguistic = pd.DataFrame(linguistic_features.tolist())


# Combine with original dataframe
df_feat = pd.concat([df.reset_index(drop=True), df_linguistic.reset_index(drop=True)], axis=1)

print(f"Extracted {len(df_linguistic.columns)} linguistic features")
print(f"\n Linguistic Features:")
print(df_linguistic.describe().T)

# TF-IDF TfidfVectorization
print("\n Creating TF-IDF Features...")

# Configure TF-IDF with optimized parameters
tfidf = TfidfVectorizer(
    max_features = 5000,
    min_df=3,
    max_df=0.8 ,
    ngram_range=(1,2),
    stop_words='english',
    sublinear_tf=True
)

# Fit and Transform
tfidf_matrix = tfidf.fit_transform(df_feat['text_clean'])
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

n_components = 100
print(f"\n Reading TF-IDF dimensions to {n_components} with SVD....")
svd = TruncatedSVD(n_components = n_components, random_state = RANDOM_STATE)
tfidf_reduced = svd.fit_transform(tfidf_matrix)

explained_variance = svd.explained_variance_ratio_.sum()
print(f"Explained Variance: {100*explained_variance: .2f}%")

# Create TF-IDF feature dataframe
tfidf_cols = [f'tfidf_{i}' for i in range(n_components)]
df_tfidf = pd.DataFrame(tfidf_reduced, columns = tfidf_cols)

# Combine all features
df_feat = pd.concat([df_feat.reset_index(drop=True), df_tfidf.reset_index(drop=True)], axis=1)

print(f"\n Total feature count: {len(df_linguistic.columns) + n_components}")

######### Modeling Strategy ############
# multi model approach with proper cross-validation and hyperparameter tuning

print("Preparing modeling dataset....\n")

# Select feature columns 
exclude_cols = [TEXT_COL, LABEL_COL, 'text_clean', 'text_length']
feature_cols = [col for col in df_feat.columns if col not in exclude_cols]

X = df_feat[feature_cols].copy()

print("Fixing categorical columns")

object_cols = X.select_dtypes(include=['object']).columns.tolist()
for col in object_cols:
    try:
        X[col] = pd.to_numeric(X[col], errors='raise')
        print(f" {col} converted to numeric")
    except:
        print(f"One hot encoding {col}")
        dummies = pd.get_dummies(X[col], prefix=col, drop_first=False)
        X = pd.concat([X.drop(col, axis=1), dummies], axis=1)

## Ensure all remaining are numeric
numeric_X = X.select_dtypes(include=[np.number])
print(f"Final numeric features: {len(numeric_X.columns)}")

## Encode Target
le_y = LabelEncoder()
y = le_y.fit_transform(df_feat[LABEL_COL])

print(f"Feature matrix shape: {numeric_X.shape}")
print(f"Target shape: {y.shape}")
print(f"Number of classes: {len(le_y.classes_)}")
print(f"\n clas mapping:")
for idx, label in enumerate(le_y.classes_):
    print(f"{idx}: {label}")

## Train test Split(Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    numeric_X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify = y
)

print(f"\n Train set size: {len(X_train)}")
print(f"Test set size: {len(X_test):,}")

## Feature scaling
print("\n Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling complete")
print("REady for modeling")

####### Evaluation Function

def evaluate_model(model, X_tr, y_tr, X_te, y_te, model_name):
    """
    Comprehensive model evaluation with multiple metrics.
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*80}")

    # Model train
    model.fit(X_tr, y_tr)

    # Predictions
    y_pred = model.predict(X_te)

    # Metrics 
    acc = accuracy_score(y_te, y_pred)
    precision = precision_score(y_te, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_te, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)

    print(f"\n Test metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # ROC-AUC (if probability predictions available)
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_te)
        try:
            roc_auc = roc_auc_score(y_te, y_pred_proba, multi_class='ovr', average='weighted')
            print(f"ROC-AUC: {roc_auc:.4f}")
        except:
            roc_auc = None
    else:
        y_pred_proba = None
        roc_auc = None

    # Cross-Validation
    print(f"\n 5-fold cross-validation")
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"CV Accuracy: {cv_scores.mean():.4f} + {cv_scores.stf():.4f}")

    # Classification Report
    print(f"\n Classification Report:")
    print(classification_report(y_te, y_pred, target_names=le_y.classes_, zero_division=0))

    # Store results
    results = {
        'model_name': model_name,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

    return results

print("Evaluation Function ready")


########## Logistic Regression (Baseline) #############

lr_model = LogisticRegression(
    max_iter=500,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight='balanced'
)

# Convert numeric classes to strings for classification_report
class_names = [str(cls) for cls in le_y.classes_]

lr_model.fit(X_train_scaled, y_train)
lr_y_pred = lr_model.predict(X_test_scaled)
lr_y_proba = lr_model.predict_proba(X_test_scaled)

print("="*80)
print(" Evaluating: Logistic Regression (Baseline)")
print("="*80)

# Test metrics
print("\n Test Metrics:")
print(f"Accuracy: {accuracy_score(y_test, lr_y_pred):.4f}")
print(f"Precision: {precision_score(y_test, lr_y_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, lr_y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_test, lr_y_pred, average='weighted', zero_division=0):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, lr_y_proba, multi_class='ovr', average='macro'):.4f}")

# Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_accuracy = cross_val_score(lr_model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
print(f"5-fold Cross-Validation: ")
print(f"CV Accuracy: {cv_accuracy.mean():.4f} + {cv_accuracy.std():.4f}")

# Fixed Classification Report
print("\n Classification Report:")
print(classification_report(y_test, lr_y_pred, target_names=class_names, zero_division=0))

lr_results = {
    'Model': 'Logistic Regression(Baseline)',
    'Accuracy': accuracy_score(y_test, lr_y_pred),
    'Precision': precision_score(y_test, lr_y_pred, average='weighted', zero_division=0),
    'Recall': recall_score(y_test, lr_y_pred, average='weighted', zero_division=0),
    'F1_Macro': f1_score(y_test, lr_y_pred, average='macro', zero_division=0),
    'F1_Weighted': f1_score(y_test, lr_y_pred, average='weighted', zero_division=0),
    'ROC_AUC': roc_auc_score(y_test, lr_y_proba, multi_class='ovr', average='macro'),
    'CV_F1_Mean': cross_val_score(lr_model, X_train_scaled, y_train, cv=cv, scoring='f1_macro').mean(),
    'CV_F1_Std': cross_val_score(lr_model, X_train_scaled, y_train, cv=cv, scoring='f1_macro').std()
}


print("\n Logistic Regression complete!")
print(lr_results)

######### Random Forest #########
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=RANDOM_STATE,
    n_jobs=1,
    class_weight='balanced'
)

# Convert numeric classes to string for classification_report
class_names = [str(cls) for cls in le_y.classes_]


rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)
rf_y_proba = rf_model.predict_proba(X_test)

print("="*80)
print("Evaluating: Random Forest")
print("="*80)

# Test Metrics
print("\n Test Metrics:")
print(f"Accuracy: {accuracy_score(y_test, rf_y_pred):.4f}")
print(f"Precision: {precision_score(y_test, rf_y_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, rf_y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_test, rf_y_pred, average='weighted', zero_division=0):.4f}")
print(f"ROC-AUC {roc_auc_score(y_test, rf_y_proba, multi_class='ovr', average='macro'):.4f}")

# Cross-Validation
cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_accuracy = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='accuracy')
print(f"\n 5-Fold Cross Validation")
print(f"CV Accuracy: {cv_accuracy.mean():.4f} + {cv_accuracy.std():.4f}")

# Fixed Classification Report
print("\n Classification Report:")
print(classification_report(y_test, rf_y_pred, target_names=class_names, zero_division=0))

rf_results= {
    'Model': 'Random Forest',
    'Accuracy': accuracy_score(y_test, rf_y_pred),
    'Precision': precision_score(y_test, rf_y_pred, average='weighted', zero_division=0),
    'Recall': recall_score(y_test, rf_y_pred, average='weighted', zero_division=0),
    'F1_Macro': f1_score(y_test, rf_y_pred, average='macro', zero_division=0),
    'F1_Weighted': f1_score(y_test, rf_y_pred, average='weighted', zero_division=0),
    'ROC_AUC': roc_auc_score(y_test, rf_y_proba, multi_class='ovr', average='macro'),
    'CV_F1_Mean': cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='f1_macro').mean(),
    'CV_F1_Std': cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='f1_macro').std()
}

print("\n Random Forest Complete!")
print(rf_results)


############# LightGBM ############
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight='balanced',
    warm_start=True
)

class_names = [str(cls) for cls in le_y.classes_]

print("Training Random Forest(fast)....")
rf_model.fit(X_train, y_train)

print("Predicting.....")
rf_y_pred = rf_model.predict(X_test)
rf_y_proba = rf_model.predict_proba(X_test)

print("\n Random Forest Results")
print(f"Accuracy: {accuracy_score(y_test, rf_y_pred):.4f}")
print(f"F1-Weighted: {f1_score(y_test, rf_y_pred, average='weighted', zero_division=0):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, rf_y_proba, multi_class='ovr', average='macro'):.4f}")

rf_results = {
    'Model': "Random Forest (Fast)",
    'Accuracy': accuracy_score(y_test, rf_y_pred),
    'F1_Weighted': f1_score(y_test, rf_y_pred, average='weighted', zero_division=0),
    'ROC_AUC': roc_auc_score(y_test, rf_y_proba, multi_class='ovr', average='macro')
}

print('\n Random Forest FAST Complete!')
print(rf_results)

########### XGBoost #########
n_classes = len(le_y.classes_)
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob' if n_classes > 2 else 'binary:logistic',
    eval_metric='mlogloss' if n_classes > 2 else 'logloss',
    num_class=n_classes if n_classes > 2 else None,
    n_estimators=100,     
    learning_rate=0.1,    # Faster
    max_depth=6,          # Shallower
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0
)

print("\n Training XGBoost (fast)..... ")
xgb_model.fit(X_train, y_train)

print("Predicting....")
xgb_y_pred = xgb_model.predict(X_test)
xgb_y_proba = xgb_model.predict_proba(X_test)

# Quick metrics
print("\n XGBoose Results:")
print(f"Accuracy: {accuracy_score(y_test, xgb_y_pred):.4f}")
print(f"F1-Weighted: {f1_score(y_test, xgb_y_pred, average='weighted', zero_division=0):.4f}")
print(f"ROC-AUC {roc_auc_score(y_test, xgb_y_proba, multi_class='ovr', average='macro'):.4f}")

xgb_results = {
    'Model': 'XGBoost (fast)',
    'Accuracy': accuracy_score(y_test, xgb_y_pred),
    'F1_Weighted': f1_score(y_test, xgb_y_pred, average='weighted', zero_division=0),
    'ROC_AUC': roc_auc_score(y_test, xgb_y_proba, multi_class='ovr', average='macro')
}

print("\n XGBoose FAST Complete!")
print(xgb_results)

############ Model Comparision ##############
models_results = []
if 'lr_results' in locals():
    models_results.append(lr_results)
if 'rf_results' in locals():
    models_results.append(rf_results)
if 'xgb_results' in locals():
    models_results.append(xgb_results)

print(f"Found {len(models_results)} model for comparision")

comparision_df = pd.DataFrame([{
    'Model': r.get('Model', 'Unknown'),
    'Test Acc': r.get('Accuracy', 0),
    'F1-Weighted': r.get('F1_Weighted', r.get('F1_Weighted', 0)),
    'ROC-AUC': r.get('ROC_AUC', 0),
    'F1_Macro': r.get('F1_Macro', 0) if 'F1_Macro' in r else 0
} for r in models_results])

# Sort by Test Accuracy
comparison_df = comparision_df.sort_values('Test Acc', ascending=False).round(4)

print('\n' + '='*60)
print("Model Comparision (Fast versions)")
print("="*60)
print(comparision_df.to_string(index=False))

# Best model
if len(comparison_df) > 0:
    best_idx = comparison_df['Test Acc'].idxmax()
    best_model = comparison_df.iloc[best_idx]
    print(f"\n🥇 BEST: {best_model['Model']}")
    print(f"   Test Acc: {best_model['Test Acc']:.4f}")
    print(f"   F1: {best_model['F1-Weighted']:.4f}")
    print(f"   ROC-AUC: {best_model['ROC-AUC']:.4f}")
else:
    print("\n❌ No model results found!")

print("\n✅ Comparison complete!")