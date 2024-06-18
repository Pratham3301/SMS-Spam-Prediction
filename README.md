# SMS Spam Classifier

This project builds an AI model to classify SMS messages as spam or legitimate (ham) using techniques like TF-IDF and classifiers such as Naive Bayes, Logistic Regression, and Support Vector Machines (SVM).

## Project Overview

The aim of this project is to detect spam messages in SMS using machine learning models. The steps include loading the dataset, preprocessing the data, vectorizing the text messages using TF-IDF, training various models, and evaluating their performance.

## Dataset

The dataset used in this project is `spam.csv`, which contains the following columns:
- `v1`: Label (ham or spam)
- `v2`: SMS message content

## Project Steps

### 1. Import Libraries

Import necessary libraries such as pandas for data manipulation, sklearn for machine learning, and seaborn and matplotlib for visualization.

### 2. Upload and Load Dataset

Use Google Colab's file upload feature to load the `spam.csv` dataset.

### 3. Preview and Preprocess Data

Preview the dataset and drop unnecessary columns. Map the labels (ham and spam) to numeric values (0 and 1).

### 4. Visualize the Class Distribution

Visualize the distribution of spam vs ham messages in the dataset.

### 5. Split Dataset

Split the dataset into training and testing sets to evaluate the model's performance.

### 6. Vectorize Text Data using TF-IDF

Convert the text messages into numerical representations using the TF-IDF vectorizer.

### 7. Train and Evaluate Naive Bayes

Train a Naive Bayes classifier on the training data and evaluate its performance on the testing data. Visualize the confusion matrix.

### 8. Train and Evaluate Logistic Regression

Train a Logistic Regression model and evaluate its performance. Visualize the confusion matrix.

### 9. Train and Evaluate SVM

Train an SVM model and evaluate its performance. Visualize the confusion matrix.

### 10. Compare Model Performance

Compare the accuracy of the Naive Bayes, Logistic Regression, and SVM models using a bar plot.

### 11. Test with New Messages

Test the models with new SMS messages to see their predictions.

## Requirements

To run this project, you need the following libraries:
- pandas
- sklearn
- matplotlib
- seaborn

Ensure you have these libraries installed before running the code.

## Running the Project

1. Open Google Colab.
2. Create a new notebook.
3. Copy and paste each code block into separate cells in the notebook.
4. Run each cell sequentially.
5. Test with your own SMS messages to see the predictions.

## Conclusion

This project demonstrates the use of machine learning models to classify SMS messages as spam or legitimate. Various models were trained and evaluated, with visualizations provided for better understanding. The final step allows for testing new SMS messages with the trained models to see their predictions.

