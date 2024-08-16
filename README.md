# Yelp Review Star Rating Prediction

In the original Udemy course "[Python for Data Science and Machine Learning Bootcamp](https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/)," the project focused solely on predicting 1-star and 5-star ratings from Yelp reviews. However, in this enhanced version, we have developed a comprehensive model that predicts all star ratings (from 1 to 5). This expansion allows for a more nuanced understanding and prediction of user sentiments across the full spectrum of ratings.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Methods](#methods)
  - [Baseline Model: Multinomial Naive Bayes](#baseline-model-multinomial-naive-bayes)
  - [SMOTE for Handling Class Imbalance](#smote-for-handling-class-imbalance)
  - [Class Weighting](#class-weighting)
  - [Text Processing with TF-IDF](#text-processing-with-tf-idf)
  - [Random Forest Classifier](#random-forest-classifier)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction
This project explores the use of various machine learning techniques to predict Yelp review star ratings from text data. While the baseline model uses a simple Multinomial Naive Bayes classifier, we also explore more sophisticated techniques like handling class imbalance with SMOTE, applying class weights, and employing a Random Forest classifier. The performance of each method is evaluated and compared to determine the most effective approach.

## Installation
To run this project, you need to have Python installed along with the necessary libraries. You can install the required libraries using pip:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes:
- pandas
- scikit-learn
- imbalanced-learn
- plotly

## Data Preparation
The data used for this project consists of Yelp reviews, which include the textual content of the reviews (`text`) and the corresponding star ratings (`stars`). We first split the data into training and testing sets:

```python
# Extract features and target from the DataFrame
X = yelp['text']
y = yelp['stars']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=101
)
```

## Methods

### Baseline Model: Multinomial Naive Bayes
The baseline model uses a Multinomial Naive Bayes classifier. The text data is converted into a matrix of token counts using `CountVectorizer`, and then TF-IDF (Term Frequency-Inverse Document Frequency) transformation is applied. The model achieved an accuracy of approximately 42.2%, but struggled particularly with predicting lower star ratings.

```python
pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

### SMOTE for Handling Class Imbalance
To address the class imbalance problem, we applied SMOTE (Synthetic Minority Over-sampling Technique). SMOTE generates synthetic samples for the minority classes, which helps the model generalize better. While SMOTE improved recall for the minority classes, the overall accuracy did not significantly improve.

```python
pipeline = ImbalancedPipeline([
    ('bow', CountVectorizer()),
    ('smote', SMOTE(random_state=101)),
    ('classifier', MultinomialNB())
])
```

### Class Weighting
An alternative approach to handling class imbalance is to assign weights to each class. We computed class weights and applied them directly to the Multinomial Naive Bayes classifier. The results showed a slight improvement in precision and recall for minority classes, but the model still struggled to predict low-star ratings accurately.

```python
class_weights = compute_class_weight('balanced', classes=classes, y=y)
pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('classifier', MultinomialNB(class_prior=class_weights))
])
```

### Text Processing with TF-IDF
We also explored the impact of text processing using TF-IDF scores instead of raw token counts. This technique improves the model's ability to distinguish between important and less important words. However, the Naive Bayes model's performance remained constrained by its inherent limitations in handling complex data distributions.

```python
pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])
```

### Random Forest Classifier
Finally, we implemented a Random Forest classifier, which is a more powerful model capable of capturing complex patterns in the data. We applied class weighting within the Random Forest model, which led to a noticeable improvement in both accuracy and the model's ability to correctly predict the 5-star reviews.

```python
pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', RandomForestClassifier(class_weight='balanced'))
])
```

## Evaluation
Across all methods, the Random Forest classifier with class weighting yielded the best performance, achieving an accuracy of approximately 43.8%. It also correctly predicted the sentiment for new reviews, demonstrating its robustness.

However, despite these improvements, the model still struggles with low-star ratings. This is a common challenge in text classification, particularly when dealing with highly imbalanced data.

## Conclusion
This project demonstrates the application of various machine learning techniques to text classification tasks, with a focus on addressing class imbalance. While each method provided some improvement, the Random Forest classifier proved to be the most effective, particularly when combined with class weighting. Nonetheless, further refinements could include exploring deep learning models or more advanced NLP techniques to better capture the nuances in the Yelp review data.
