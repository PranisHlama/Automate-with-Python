# Fake News Detector Project in AI

Fake news is misleading or false information that is circulated as news. It is often difficult to distinguish between fake and real news, and it isn’t until the situation gets blown out of proportion that it comes to light. The spreading of fake news becomes especially dangerous during times like elections or pandemic situations. Fake rumours and misinformation that pose harm to human lives are threatening to people and the society. Fake news needs to be detected and prevented early, before it causes panic and spreads to a large number of people.

**Project Idea:** For this very interesting project, you will build a fake news detector, you can use the Real and Fake News dataset  You can use a pre-trained machine learning model called BERT to perform this classification. BERT is a Natural Language Processing (NLP) model that has been made open-source. You can load BERT into Python and just add one additional output layer for your text classification task.

Tools and Libraries: Python, BERT

Dataset: Fake News Dataset | IEEE DataPort

# Fake News Detector Using BERT

## 1. Understand the Problem

- Learn what fake news is and why it is harmful.
- Understand the difference between binary text classification and multiclass classification.
- Define the project objective:
  - Input: News article text
  - Output: Real or Fake

---

## 2. Gather and Understand the Dataset

- Download the Fake News Dataset.
- Examine the dataset structure.
- Identify:
  - Features (title, text, etc.)
  - Target labels (Real/Fake)
- Check dataset size and class distribution.
- Understand any data quality issues.

---

## 3. Perform Exploratory Data Analysis (EDA)

- Inspect sample records.
- Check for missing values.
- Analyze class imbalance.
- Calculate average article length.
- Visualize label distribution.
- Identify common words and phrases.

---

## 4. Clean and Preprocess the Data

- Remove duplicate records.
- Handle missing values.
- Convert text to a consistent format.
- Remove unwanted characters and noise.
- Normalize text where necessary.
- Prepare text for tokenization.

---

## 5. Learn About BERT

- Understand Transformer architecture at a high level.
- Learn how BERT works.
- Study:
  - Tokenization
  - Attention mechanism
  - Contextual embeddings
  - Pre-training and fine-tuning
- Understand why BERT performs well for NLP tasks.

---

## 6. Prepare Data for BERT

- Tokenize news articles using BERT tokenizer.
- Convert tokens into numerical representations.
- Handle sequence length limitations.
- Create attention masks.
- Prepare labels for training.

---

## 7. Split the Dataset

- Create training dataset.
- Create validation dataset.
- Create testing dataset.
- Ensure balanced distribution of classes.

---

## 8. Load a Pre-trained BERT Model

- Select an appropriate BERT variant.
- Configure the model for binary classification.
- Understand the classification layer added on top of BERT.

---

## 9. Train the Model

- Feed training data into the model.
- Monitor:
  - Training loss
  - Validation loss
  - Accuracy
- Tune hyperparameters when necessary.
- Prevent overfitting.

---

## 10. Evaluate Model Performance

- Measure:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Generate a confusion matrix.
- Analyze incorrect predictions.

---

## 11. Test With Unseen News Articles

- Provide completely new articles.
- Observe prediction quality.
- Analyze strengths and weaknesses of the model.

---

## 12. Save the Trained Model

- Save model weights.
- Save tokenizer configuration.
- Store metadata needed for future inference.

---

## 13. Build an Inference Pipeline

- Accept news text as input.
- Preprocess the text.
- Convert text into BERT inputs.
- Generate prediction.
- Display classification result.

---

## 14. Create a User Interface (Optional)

Possible interfaces:

- Command Line Interface (CLI)
- Flask Web Application
- Django Web Application
- Streamlit Dashboard
- REST API

---

## 15. Analyze Project Limitations

- False positives and false negatives.
- Bias in training data.
- Generalization to new topics.
- Detection of satire or opinion pieces.
- Language limitations.

---

## 16. Future Enhancements

- Multi-language fake news detection.
- Real-time news verification.
- Social media integration.
- Explainable AI predictions.
- Fact-checking integration.
- Advanced transformer models beyond BERT.

---

## Learning Outcomes

By completing this project, you will learn:

- Natural Language Processing (NLP)
- Text preprocessing
- Transformer models
- BERT architecture
- Binary text classification
- Model evaluation
- AI model deployment

# Explanation of the code
The dataset contains both fake and real news articles. 
The label distribution shows whether the dataset is balanced or imbalanced.

The average article length was calculated using word count. 
This helps us understand whether fake and real articles differ in writing length.

Common words and phrases were extracted after removing stop words.
This helps identify repeated vocabulary patterns in fake and real news articles.