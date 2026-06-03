# 3. Resume Parser AI Project

## Overview

Recruiters spend a significant amount of time reviewing resumes to identify the most suitable candidates for a job position. Since a single vacancy may receive hundreds of applications, this process is often automated using techniques such as keyword matching. However, traditional keyword-based systems may fail to accurately assess a candidate's qualifications.

A Resume Parser AI can leverage Artificial Intelligence (AI) and Machine Learning (ML) techniques to analyze resumes more intelligently. Instead of relying solely on keyword matching, the system can understand the context of skills and experiences, helping recruiters identify the most qualified candidates while filtering out resumes filled with irrelevant keywords.

## Project Idea

This project involves building an AI-powered resume parser using the Resume Dataset available on Kaggle. The dataset contains two primary columns:

- **Job Title**
- **Resume Information**

The resume data is stored as text and requires preprocessing before it can be used for machine learning tasks.

## Data Preprocessing

The preprocessing stage may include:

- Converting text to lowercase
- Removing punctuation and special characters
- Removing stop words
- Tokenization
- Lemmatization or stemming

The **NLTK (Natural Language Toolkit)** Python library can be used to perform these preprocessing tasks efficiently.

## Model Development

After preprocessing, a clustering algorithm can be developed to:

- Group related skills and keywords together
- Identify domain-specific competencies
- Understand contextual relationships between words
- Detect relevant candidate qualifications beyond simple keyword matching

The model should consider words that are similar in meaning and context rather than exact keyword matches.

## Resume Scoring System(Incomplete)

A scoring mechanism can be implemented to evaluate candidate suitability for a specific job role. The system can assign a score ranging from:

- **0** – Least favorable candidate
- **10** – Most favorable candidate

The score can be calculated based on:

- Skill relevance
- Experience alignment
- Contextual similarity between resume content and job requirements
- Domain-specific competencies

## Expected Outcome

The final system should be capable of:

- Automatically parsing resumes
- Extracting important skills and qualifications
- Clustering related competencies
- Ranking candidates based on suitability
- Reducing manual screening effort for recruiters

## Conclusion

The Resume Parser AI Project serves as an excellent introductory project for learning Artificial Intelligence, Natural Language Processing (NLP), and Machine Learning. It demonstrates how AI can be applied to solve real-world recruitment challenges by automating resume screening and candidate evaluation.

## How to run project:
To get the actual CSV file:
```
sudo apt install git-lfs
git lfs install
git lfs pull
```
After that, **Resume.csv** should become the real 54 MB CSV file.

## Install Spacy + model:
```
pip install spacy
python -m spacy download en_core_web_sm
```