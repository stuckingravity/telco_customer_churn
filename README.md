# telco_customer_churn
End-to-end churn prediction project for a subscription-based business, covering data preprocessing, feature engineering, model training and business-driven retention insights using Python and machine learning.
# Subscription Churn Prediction using Machine Learning

## 1. Business Context

Customer churn is one of the most critical challenges in subscription-based industries such as telecommunications, streaming platforms, and SaaS services. Since revenue depends on recurring payments, retaining existing customers is often significantly more cost-effective than acquiring new ones.

This project develops an end-to-end machine learning pipeline to predict customer churn and identify high-risk users, enabling data-driven retention strategies and proactive intervention.

The main objectives are:

- Compare different machine learning models for churn prediction
- Evaluate model performance under class imbalance
- Identify key drivers of churn
- Provide actionable business insights for retention strategy

---

## 2. Problem Definition

Customer churn prediction is formulated as a supervised binary classification problem:

- Target variable:  
  - 1 → Customer churned  
  - 0 → Customer retained  

Each customer is represented by demographic attributes, service usage features, and contract-related information.

The goal is to learn a predictive function:

f(x) → {0,1}

that accurately identifies customers likely to churn.

---

## 3. Dataset

The dataset is based on a real-world subscription business scenario (Telco Customer Churn dataset from Kaggle), containing:

- 7,043 customer records
- 21 features
- Demographic information
- Contract type
- Payment details
- Service subscriptions
- Churn label

Although originally from telecommunications, the modeling framework generalizes directly to streaming and SaaS subscription platforms.

---

## 4. Methodology

### 4.1 End-to-End Pipeline

The workflow includes:

- Data cleaning
- Feature preprocessing
- Class imbalance handling (SMOTE)
- Model training
- Model evaluation
- Business interpretation

---

### 4.2 Data Preprocessing

- Converted monetary variables to numeric format
- Removed non-informative identifiers (customerID)
- Binary encoding for two-class categorical variables
- One-hot encoding for multi-class categorical variables
- Standardization of numerical features using z-score normalization

---

### 4.3 Handling Class Imbalance

Customer churn datasets are typically imbalanced.

To improve minority-class detection, SMOTE (Synthetic Minority Oversampling Technique) was applied **only on the training set** to avoid information leakage.

This improves recall for churned customers while preserving test set integrity.

---

## 5. Models Implemented

Three widely used classification models were evaluated:

### Logistic Regression
- Interpretable linear baseline
- Optimized via cross-entropy loss
- Strong recall performance

### Decision Tree (CART, Gini impurity)
- Captures non-linear feature interactions
- Interpretable rule-based splits
- Prone to overfitting

### Random Forest
- Ensemble of decision trees
- Reduces variance via bagging
- Strong generalization performance

---

## 6. Evaluation Metrics

Due to class imbalance, accuracy alone is insufficient.

Models were evaluated using:

- Precision
- Recall
- F1-score
- ROC-AUC

Special emphasis was placed on churn recall to minimize false negatives (missed churners).

---

## 7. Results

| Model | Precision | Recall | F1 | Accuracy | ROC-AUC |
|--------|----------|--------|----|----------|----------|
| Logistic Regression | 0.50 | 0.80 | 0.62 | 0.74 | 0.84 |
| Decision Tree | 0.47 | 0.54 | 0.50 | 0.72 | 0.66 |
| Random Forest | 0.57 | 0.58 | 0.57 | 0.77 | 0.82 |

### Key Findings

- Logistic Regression achieved the highest ROC-AUC (0.84) and recall (0.80), making it effective at identifying churners.
- Decision Trees showed signs of overfitting.
- Random Forest provided the most balanced performance.

---

## 8. Business Insights

- Customers with shorter tenure show significantly higher churn probability.
- Contract type strongly influences churn risk.
- Payment method and monthly charges are important churn indicators.

Potential applications include:

- Targeted retention campaigns
- Personalized offers
- Early churn intervention
- Revenue forecasting
- Customer segmentation

---

## 9. Limitations

- Dataset lacks longitudinal behavioral usage data.
- Advanced models (e.g., Gradient Boosting, Neural Networks) were not explored.
- SMOTE may introduce synthetic noise.
- External validation on other subscription industries would improve robustness.

---

## 10. Future Improvements

- Add time-series behavioral features (e.g., activity recency)
- Implement gradient boosting (XGBoost / LightGBM)
- Deploy model as a REST API
- Implement model monitoring & drift detection
- Integrate SQL-based feature engineering pipeline

---

## 11. Tech Stack

- Python
- Pandas
- Scikit-learn
- SMOTE (imbalanced-learn)
- Matplotlib / Seaborn
