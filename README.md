# Predicting Amazon Movie Review Ratings Using an Ensemble Model and Custom Features
This project focuses on predicting star ratings for Amazon movie reviews. The goal was to create a machine learning model capable of accurately predicting ratings using various engineered features and an ensemble of classifiers. The final model utilizes Random Forest, Stochastic Gradient Descent (SGD), and Ordinal Logistic Regression classifiers combined in a weighted soft voting ensemble. 


## Key Components
- **Feature Engineering**: Leveraged review helpfulness scores, sentiment polarity, TF-IDF, review age, text length, and composite features to improve model accuracy.
- **Modeling**: Implemented Random Forest, SGD Classifier, and Ordinal Logistic Regression models.
- **Ensemble Learning**: Combined model predictions with a weighted soft voting approach for improved performance.
- **Evaluation**: Used classification accuracy and a confusion matrix to assess model performance. Achieved a 58.76% accuracy for the ensemble model overall, higher than any individual model's accuracy.

## Feature Engineering
The following features were engineered to capture various aspects of review data:
- **Helpfulness Score**: Ratio of helpfulness numerator to denominator.
- **TF-IDF**: Textual features with the top 200 terms, reduced via Truncated SVD.
- **Sentiment Polarity**: Basic sentiment analysis score using TextBlob.
- **Composite Features**: Combinations such as sentiment-helpfulness and review age-helpfulness


## Model Overview
The project uses three primary models:
- **Random Forest**: Captures non-linear interactions and has robust feature importance scores.
- **SGD Classifier**: Efficient for large datasets, especially for sparse text data.
- **Ordinal Logistic Regression**: Suitable for ordinal data like star ratings.

## Ensemble Model
The final model combines the three classifiers in a weighted soft voting ensemble with weights as follows:
- **Random Forest**: 0.5
- **SGD Classifier**: 0.2
- **Ordinal Logistic Regression**: 0.3
This ensemble approach allowed the model to leverage the unique strengths of each classifier for more stable and accurate predictions.
