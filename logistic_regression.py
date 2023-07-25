'''
Logistic Regression

Logistic regression allows us to solve classification problems, where we are trying to predict 
discrete categories (i.e binary classification, ham/spam, etc). Logistic regressions utilizes the
Sigmoid function which takes any value and outputs to either 0 or 1.

This script will use logistic regression on a mock advertising dataset and determine whether or not
a particular user click on an Ad. The model will be used to predict whether or not a user will click
on an add given their features.
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

class LogisticRegressor:
    def __init__(self) -> None:
        self.data = pd.read_csv('./data/advertising.csv')
        self.model = None
        self.X_test = None
        self.y_test = None
    
    def check_data(self):
        """
        Key features:
        0   Daily Time Spent on Site
        1   Age 
        2   Area Income
        3   Daily Internet Usage
        4   Ad Topic Line
        5   City
        6   Male
        7   Country
        8   Timestamp
        9   Clicked on Ad 

        """
        print(f'{self.data.head(5)}\n\n')  # print first 5 elements
        print(f'{self.data.describe()}\n\n')  # check stats
        print(f'{self.data.info()}\n\n')  # check features

    def eda(self):
        """
        Exploratory Data Analysis

        """
        sns.set_style('whitegrid')
        sns.pairplot(self.data, hue='Clicked on Ad', palette='bwr', )
        sns.displot(data=self.data, x='Age')  # Age distribution
        sns.jointplot(data=self.data, x='Age', y='Area Income')  # Income vs Age
        sns.jointplot(data=self.data, x='Age', y='Daily Time Spent on Site', kind='kde', 
                      fill=True, color='red')  # Age vs Time on Site
        sns.jointplot(data=self.data, x='Daily Time Spent on Site', y='Daily Internet Usage')

        plt.show()
    
    def train(self):
        """
        Train the model

        """
        # Split data into features and labels
        X = self.data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 
                       'Male']]  # Only train on a few key features
        y = self.data['Clicked on Ad']

        # Split data into training and testing subsets
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.3, 
                                                                      random_state=42)

        # Train the model
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)

    def eval(self):
        """
        Evaluate the model

        The model shows good precision and recall for the given size of the dataset. There were
        only a few FP and TN that were missed, but majority of the data points were predicted 
        correctly.
        """
        
        # Confusion Matrix and Classification Report
        predictions = self.model.predict(self.X_test)
        print(confusion_matrix(self.y_test, predictions))
        print(classification_report(self.y_test, predictions))

    def interpret(self):
        """
        Interpret the data

        
        """
        # Construct a sample dataframe with average values for the features
        df = pd.DataFrame([[1000, 20, 80000, 1000, 1], 
                           [1000, 20, 80000, 1000, 0], 
                           [70, 50, 40000, 122, 0]], 
                          columns=['Daily Time Spent on Site', 'Age', 'Area Income',
                                   'Daily Internet Usage', 'Male'])
        predictions = self.model.predict(df)
        
        # Print
        for sample_num, pred in enumerate(predictions):
            print(f'{sample_num}: Ad Clicked = {pred}')

if __name__ == '__main__':
    lr = LogisticRegressor()
    lr.check_data()
    lr.eda()
    lr.train()
    lr.eval()
    lr.interpret()
