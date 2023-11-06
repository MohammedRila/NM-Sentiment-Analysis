                                        Sentiment Analysis for Marketing in Airline Tweets

Overview:
This project performs sentiment analysis on airline-related tweets to gain insights into customer opinions and improve marketing strategies.

Table of Contents:
- Introduction
- Problem Statement
- Design Thinking Process
- Project Phases
- Libraries and Tools
- Data Source
- Data Preprocessing
- Text Vectorization
- Machine Learning Model
- Results and Visualizations
- README Instructions

Introduction:
This section provides a brief overview of the project and its objectives. It explains the purpose of sentiment analysis on airline tweets for marketing insights.

Problem Statement:
Describes the problem to be solved through sentiment analysis, which is to gain insights into customer opinions and improve marketing strategies in the airline industry.

Design Thinking Process:
Explains the design thinking process, including defining the problem, selecting libraries and tools, data preprocessing, visualization, and model training.

Project Phases:
Outlines the different phases of the project, including data loading, preprocessing, text vectorization, machine learning model implementation, and visualizations.


Libraries and Tools:
Lists the libraries and tools used in the project, including Python, Pandas, Matplotlib, Seaborn, NLTK, and Scikit-Learn.

Data Source:
Indicates the source of the dataset, which is "Tweets.csv."

Data Preprocessing:
Details the data preprocessing steps, such as removing special characters, single characters, multiple spaces, 'b' prefixes, and converting text to lowercase.

Text Vectorization:
Explains how text data is transformed into numerical vectors using TF-IDF vectorization and how stop words are removed.

Machine Learning Model:
Describes the use of a Random Forest Classifier for sentiment classification and how data is split into training and testing sets. It also includes information about model performance evaluation.

Results and Visualizations:
Lists the visualizations used in the project, such as pie charts displaying airline and sentiment distribution and a bar plot showing the count of sentiment categories for each airline.

                                                                       README Instructions
Running the Code:
1. Ensure you have the necessary libraries and tools installed, as listed in the "Libraries and Tools" section.

2. Download the dataset from the provided link: [Tweets.csv](https://www.kaggle.com/datasets/crowdflower/twitterairlinesentiment).

3. Clone this project repository or download the Python code.

4. Open the Python script and update the data file path if necessary.

5. Run the script to perform sentiment analysis on the airline tweets.

 Dependencies:
- Python
- Pandas
- Matplotlib
- Seaborn
- NLTK
- Scikit-Learn

The code you provided appears to be a Python script for performing sentiment analysis on airline-related tweets using machine learning techniques. I'll explain each step of the code for better understanding:

Step 1: Import Libraries

import numpy as np
import pandas as pd
import re 
import nltk 
import matplotlib.pyplot as plt  
%matplotlib inline  

In this step, the necessary Python libraries are imported. numpy and pandas are used for data manipulation, re for regular expressions, nltk for natural language processing, and matplotlib for data visualization.

Step 2: Load Data

airline_tweets = pd.read_csv(r'D:\Tweets.csv')
Here, the code reads a CSV file containing airline-related tweets and stores the data in a Pandas DataFrame called airline_tweets.

Step 3: Adjust Plot Size

plot_size = plt.rcParams["figure.figsize"]
print(plot_size[0])
print(plot_size[1]) 
plot_size[0] = 8
plot_size[1] = 6
plt.rcParams["figure.figsize"] = plot_size

This step modifies the default plot size of Matplotlib to make the visualizations larger by changing the figure.figsize parameters.

Step 4: Create Pie Charts and a Bar Plot

airline_tweets.airline.value_counts().plot(kind='pie', autopct='%1.0f%%')
airline_tweets.airline_sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["brown", "gold", "blue"])
These lines of code create pie charts to visualize the distribution of airlines and sentiments in the tweet dataset.


airline_sentiment = airline_tweets.groupby(['airline', 'airline_sentiment']).airline_sentiment.count().unstack()
airline_sentiment.plot(kind='bar')
Here, a bar plot is created to show the distribution of sentiments for each airline. This involves grouping the data by both the airline and sentiment and then plotting the results.


import seaborn as sns
sns.barplot(x='airline_sentiment', y='airline_sentiment_confidence', data=airline_tweets)

This code uses Seaborn to create a bar plot, but it seems there's a typo in the column names, which should be fixed for it to work properly.

Step 5: Text Preprocessing

features = airline_tweets.iloc[:, 10].values
labels = airline_tweets.iloc[:, 1].values
processed_features = []

This step prepares the text data for machine learning. It extracts the tweet content as features and the sentiment labels as labels.


# Text preprocessing using regular expressions
for sentence in range(0, len(features)):
    # Remove special characters, single characters, multiple spaces, 'b' prefixes, and convert to lowercase
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))
    processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
    processed_feature = re.sub(r'^b\s+', '', processed_feature)
    processed_feature = processed_feature.lower()
    processed_features.append(processed_feature)

This code cleans and preprocesses the text data. It removes special characters, single characters, multiple spaces, and converts the text to lowercase. The preprocessed text is stored in the processed_features list.

Step 6: Text Vectorization

from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer 
vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()

In this step, text data is transformed into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. It also removes common English stopwords. The resulting features are stored in the processed_features variable.

Step 7: Machine Learning Model

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)
The dataset is split into training and testing sets for machine learning model evaluation.


from sklearn.ensemble import RandomForestClassifier
text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)
predictions = text_classifier.predict(X_test)

A Random Forest classifier is trained on the processed features and labels. The model is used to make predictions on the test data.


from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, predictions))
print('accuracy score', accuracy_score(y_test, predictions))

Finally, the code prints a confusion matrix and the accuracy score to evaluate the performance of the machine learning model.

Overall, this code performs the following tasks: data loading, data visualization, text preprocessing, text vectorization, and machine learning model training and evaluation for sentiment analysis of airline-related tweets. However, there are some issues in the code, such as the typo in the Seaborn barplot and the need to handle missing or incorrect data in the dataset.


This README file provides instructions for setting up and running the sentiment analysis code on airline tweets. Follow the steps above to analyze the data and gain valuable marketing insights.
