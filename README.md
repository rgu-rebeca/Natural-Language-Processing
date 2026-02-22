# Natural-Language-Processing

🧠 Sentiment Analysis on Amazon Reviews

Traditional Machine Learning vs. Transfer Learning

Author: Rebeca (Ruijia) Gu
Date: 2026-02-22

## 📌 Project Overview

This project implements and compares different Natural Language Processing (NLP) approaches for binary sentiment classification on Amazon product reviews.

Two product categories were analyzed:

🚗 Automotive Reviews

👶 Baby Products Reviews

The objective is to classify reviews as positive or negative based on their textual content and star ratings.

The project explores:

- Classical Machine Learning (TF-IDF + SVM / Logistic Regression)

- Data augmentation (Back Translation)

- Transfer Learning (DistilBERT)
The focus is not only on performance, but also on cost–benefit trade-offs, computational constraints, and practical deployment considerations.

## 📂 Datasets

**Source**: Amazon Review Data Repository (UCSD)
http://jmcauley.ucsd.edu/data/amazon/

Each dataset contains:

-Review text

-Star rating (1–5)

-Label Transformation

Ratings were converted into binary labels:

 -Low ratings → Negative (class 0)

 -High ratings → Positive (class 1)

Both datasets show strong class imbalance, with significantly more positive reviews.

## 🔍 Exploratory Data Analysis (EDA)

Performed analyses include:

-Rating distribution

-Class distribution

-Most frequent n-grams

-Word clouds (positive vs negative)

-Word2Vec embedding visualization (Automotive dataset)

Key insight:

-Positive reviews dominate.

-Clear lexical differences exist between positive and negative reviews.

-Stopwords dominate raw frequency analysis, confirming the need for preprocessing.
## 🧹 Preprocessing Pipeline

Implemented steps:

-Lowercasing

-Removal of punctuation / non-alphabetic characters

-Stopword removal (special handling for "no")

-Tokenization

-Removal of empty reviews

In the Baby dataset, 100,000 samples were loaded and cleaned before modeling.

## 🔢 Feature Engineering
TF-IDF Vectorization

Configuration (Baby dataset):

-max_df=0.95

-min_df=3

-max_features=2500

-ngram_range=(1,2)

Trigrams were tested but introduced noise.

Chi-square feature analysis confirmed strong discriminative vocabulary terms.

## 🤖 Models Implemented
## 🚗 Automotive Dataset
### Baseline Model

**TF-IDF + Support Vector Machine (SVM)**

-Hyperparameter tuning with GridSearchCV

-Optimized regularization parameter C

-~80% accuracy

Issue identified:

-Strong bias toward majority class

-Many negative reviews misclassified

## Data Augmentation – Back Translation

Applied only to negative class to reduce imbalance:

English → Spanish → English
Using MarianMT transformer models

Due to computational constraints:

-Initially planned: 5000 samples

-Final implementation: 2000 samples

Result:

-Slight improvement in validation score

-Limited improvement in minority recall

-High computational cost relative to performance gain


## 👶 Baby Dataset
### 1️⃣ Logistic Regression + TF-IDF

Configuration:

-Class weight = balanced

-GridSearch over C values

-5-fold cross-validation

-Optimized using f1_macro

Additional step:

-Probability threshold tuning to improve precision for class 0

**Results:**

-ROC-AUC: 0.909

-Strong performance for majority class

-Minority class still challenging

### 2️⃣ Transfer Learning – DistilBERT

**Model: distilbert-base-uncased**

Reasons for selection:

-40% smaller than BERT-base

-Retains ~97% of BERT performance

-More computationally efficient

Training details:

-Max length: 128

-Batch size: 16

-Epochs: 3

-GPU training (~2 hours)

-Best model selected by f1_macro

**Results:**

-Significant improvement in minority class precision

-Higher macro F1 score

-Better contextual understanding

However:

-Slightly lower ROC-AUC than Logistic Regression

-Much higher computational cost

## 📊 Traditional ML vs Transfer Learning

| Aspect | Logistic Regression | DistilBERT |
|--------|--------------------|------------|
| Computational Cost | Low | High |
| Training Time | Fast | ~2 hours (GPU) |
| Interpretability | High | Low |
| Minority Class Precision | Low | Much Higher |
| Context Understanding | Limited | Strong |
| Deployment Simplicity | Easy | Complex |

## ⚠️ Challenges

- Severe class imbalance  
- Minority class detection difficulty  
- High cost of back translation  
- GPU dependency for transformers  
- Data alignment issues during augmentation  

---

## 🧠 Key Conclusions

- TF-IDF + Linear Models are strong baselines.  
- Class imbalance heavily affects minority recall.  
- Back translation adds cost but limited gain for classical models.  
- DistilBERT significantly improves minority class performance.  
- Model choice depends on computational resources and business needs.  

---

## 🚀 Future Work

- Explore RoBERTa / ALBERT  
- Test lighter augmentation methods  
- Compare augmentation vs class weighting  
- Try ensemble approaches  
- Deploy model and monitor real-world drift  

---

## 🛠 Technical Stack

- Python 3  
- pandas, numpy, nltk, re  
- scikit-learn  
- transformers, torch, datasets  
- matplotlib, wordcloud  
- Google Colab (GPU)  

---

## 🎯 Final Reflection

This project demonstrates that increasing model complexity does not automatically guarantee proportional improvements.

In resource-constrained environments, classical machine learning methods provide an excellent performance–cost balance.

Transfer learning offers superior contextual understanding, but at significantly higher computational expense.

Careful evaluation of trade-offs is essential in applied NLP systems.
