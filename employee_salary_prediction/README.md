# Employee Salary Prediction Application

A machine learning-powered Streamlit web application that predicts employee salaries based on various factors such as age, years of experience, and education level. The application also provides comprehensive exploratory data analysis (EDA) and salary analysis visualizations.

## 🎯 Overview

This application uses a **Random Forest Regressor** model to predict employee salaries with high accuracy. It features an interactive dashboard with multiple sections for data exploration, analysis, and real-time salary predictions.

### Key Features
-  **Interactive Dashboard** - Explore datasets and visualizations
-  **Exploratory Data Analysis (EDA)** - Comprehensive data statistics and distributions
-  **Salary Analysis** - Insights into salary trends by gender, education, and experience
-  **Machine Learning Model** - Random Forest-based salary prediction
-  **Beautiful Visualizations** - Plotly-powered interactive charts
-  **Real-time Predictions** - Instant salary predictions with custom inputs

---

##  Project Structure
employee_salary_prediction/ 
├── app.py # Main Streamlit application 
├── employee_salary_dataset.csv # Primary dataset for training 
├── archive.csv # Historical/archived dataset 
├── salary_model.pk1 # Pre-trained Random Forest model 
└── README.md # Project documentation

### File Descriptions

| File | Size | Description |
|------|------|-------------|
| `app.py` | 13.5 KB | Main application with data processing, model training, and UI |
| `employee_salary_dataset.csv` | 19.4 KB | Training dataset with 50+ employee records |
| `archive.csv` | 809 B | Archived historical data |
| `salary_model.pk1` | 7 MB | Pre-trained Random Forest model (500 estimators) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/PranisHlama/Automate-with-Python.git
   cd Automate-with-Python/employee_salary_prediction

2. **Create a virtual environment (optional but recommended)**
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. **Install dependencies**
pip install -r requirements.txt

4. **Running the Application**
```
streamlit run app.py
```
The application will open in your default browser at http://localhost:8501

# Application Features

## 1. Dataset Section

- View complete employee dataset
- Dataset shape and structure
- Statistical summary:
  - Mean
  - Median
  - Standard deviation
  - Minimum
  - Maximum
- Missing values analysis
- Duplicate record detection

## 2. EDA (Exploratory Data Analysis) Section

### Age Distribution
- Box plots
- Histograms
- Median line visualization

### Gender Analysis
- Frequency distribution
- Percentage breakdowns

### Education Level Analysis
- Distribution across:
  - Bachelor's
  - Master's
  - PhD

### Experience Distribution
- Years of experience patterns

### Correlation Matrix
- Heatmap showing feature relationships

### Scatter Matrix
- Multi-dimensional relationship visualization

## 3. Salary Analysis Section

- Average salary by gender
- Average salary by education level
- Salary trends across experience levels
- Comparative bar charts with hover details

## 4. Prediction Section

### Interactive Input Form
- Age (19-65 years)
- Years of Experience (0-40 years)
- Education Level:
  - Bachelor's
  - Master's
  - PhD

### Outputs
- Real-time salary predictions

### Model Performance Metrics
- R² Score
- RMSE (Root Mean Squared Error)
- Cross-validation Score

### Visualizations
- Predicted vs. Actual salary scatter plot
- Feature importance visualization

---

# Machine Learning Model

## Model Details

| Parameter | Value |
|------------|--------|
| Algorithm | Random Forest Regressor |
| Number of Estimators | 500 |
| Validation Method | K-Fold Cross Validation (10 splits) |
| Test Size | 20% |
| Random State | 90 |

## Model Performance Metrics

| Metric | Performance |
|----------|-------------|
| R² Score | ~85-90% |
| Cross-validation Score | High accuracy across folds |
| RMSE | Low error margin |

## Features Used for Prediction

### Age
Employee's current age

### Years of Experience
Professional experience

### Education Level (Encoded)
- Bachelor's (baseline)
- Master's (binary encoded)
- PhD (binary encoded)

---

# Data Structure

## Input Dataset Columns

| Column | Description |
|----------|-------------|
| Age | Employee age |
| Gender | Male/Female |
| Education Level | Bachelor's/Master's/PhD |
| Years of Experience | Professional experience |
| Job Title | Employee's position |
| Salary | Target variable (annual salary) |

## Data Preprocessing

- **Missing Values:** Removed records with null values
- **Duplicates:** Removed duplicate records
- **Encoding:** One-hot encoding for categorical variables
- **Feature Selection:** Selected relevant features, excluded Job Title and Gender

---

# Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.8+ | Core language |
| Streamlit | Latest | Web framework |
| Pandas | Latest | Data manipulation |
| NumPy | Latest | Numerical computing |
| Scikit-learn | Latest | Machine learning |
| Plotly | Latest | Interactive visualizations |
| Joblib | Latest | Model serialization |

---

# Future Enhancements

## 1. Enhanced Model Architecture

- Implement multiple regression models:
  - XGBoost
  - LightGBM
  - Neural Networks
- Add model comparison dashboard
- Implement hyperparameter tuning with GridSearchCV
- Add ensemble methods combining multiple models
- Implement model versioning and A/B testing

## 2. Data & Features

- Add more features:
  - Location
  - Department
  - Years at company
  - Performance rating
- Implement feature engineering pipeline
- Add categorical features such as job title and department predictions
- Include salary history tracking and trends
- Add market benchmarking data

## 3. User Interface

- Custom CSS styling for enhanced appearance
- Dark/Light theme toggle
- Export functionality (PDF/CSV reports)
- Multi-language support
- Data upload capability for custom predictions
- Interactive dashboard with drill-down capabilities

## 4. Predictions & Insights

- Add confidence intervals for predictions
- Implement SHAP values for model explainability
- Add "What-if" analysis scenarios
- Generate salary recommendation reports
- Add salary range instead of point prediction
- Implement fairness analysis (gender/education bias detection)

## 5. Data Management

- Database integration (PostgreSQL/MongoDB)
- Real-time data updates
- Historical comparison and trend analysis
- Data versioning and audit trails
- Automated data validation and quality checks

## 6. Performance & Scalability

- Model caching and optimization
- API endpoint creation (FastAPI)
- Batch prediction capabilities
- Real-time model monitoring
- Scalable deployment on cloud platforms

## 7. Documentation & Testing

- Unit tests for model components
- Integration tests for data pipeline
- API documentation with Swagger/OpenAPI
- User guides with video tutorials
- Technical architecture documentation

## 8. Advanced Analytics

- Predictive salary progression forecasts
- Anomaly detection for unusual salaries
- Clustering analysis to identify employee groups
- Retention risk prediction
- Salary gap analysis and recommendations

## 9. Deployment & DevOps

- Dockerization for easy deployment
- CI/CD pipeline with GitHub Actions
- Automated model retraining schedule
- Cloud deployment:
  - AWS
  - GCP
  - Azure
- Monitoring and alerting system

## 10. Security & Compliance

- User authentication and authorization
- Data encryption (in transit and at rest)
- Audit logging
- GDPR/CCPA compliance
- Data anonymization options