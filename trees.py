'''
Decision Trees & Random Forests

Decision trees and random forests are both algorithms used for classification and regression.
Trees are similar to flowcharts and make predictions based off of input features. Each branch
of the tree represents possible outcomes that decison can take. They are simple, but can be prone
to overfitting

Random Forests solve the overfitting issue by creating an ensemble of multiple decision trees. Each
tree in the forest is trained independetly and can make predictions on its own. Each tree "votes"
for the final outcome, and most popular prediction becomes the final result. These are more robust
and better for larger datasets.

The following script trains both models and compares them against each other. The script will use
public data from LendingClub to help determine if a borrower will repay a loan or not.

'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

class Trees:
    def __init__(self) -> None:
        self.data = pd.read_csv('./data/loan_data.csv')
        self.dtree = None
        self.rfc = None
        self.X_test = None
        self.y_test = None
    
    def check_data(self):
        """
        Check data
        """
        print(f'{self.data.head(5)}\n\n')  # print first 5 elements
        print(f'{self.data.describe()}\n\n')  # check stats
        print(f'{self.data.info()}\n\n')  # check features

    def eda(self):
        """
        Exploratory Data Analysis

        """
        sns.displot(data=self.data, x='fico', hue='credit.policy')  # FICO score for credit policy
        sns.displot(data=self.data, x='fico', hue='not.fully.paid')  # FICO for paid result
        sns.countplot(data=self.data, x='purpose', hue='not.fully.paid')
        sns.jointplot(data=self.data, x='fico', y='int.rate')  # FICO vs Interest rate
        sns.lmplot(data=self.data, x='fico', y='int.rate', hue='credit.policy', col='not.fully.paid')
        plt.show()
    
    def clean(self):
        """
        This step will convert the purpose feature into dummy variables. This is because purpose
        is a categorical feature and we need to break it up in order to train the model.

        """
        cat_feats = ['purpose']
        self.data = pd.get_dummies(data=self.data, columns=cat_feats, drop_first=True)
        print(self.data.info())
        print(self.data.head())


    def train(self):
        """
        Train the model

        """
        # Split data into features and labels
        X = self.data.drop('not.fully.paid', axis=1)
        y = self.data['not.fully.paid']  # the label we want to predict

        # Split data into training and testing subsets
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.3, 
                                                                      random_state=101)
        
        # Train decision tree
        self.dtree = DecisionTreeClassifier()
        self.dtree.fit(X_train, y_train)

        # Train random forest
        self.rfc = RandomForestClassifier()
        self.rfc.fit(X_train, y_train)

    def eval(self):
        """
        Evaluate the model

        The random forest slighty outperforms the tree

        """
        dtree_predictions = self.dtree.predict(self.X_test)
        rfc_predictions = self.rfc.predict(self.X_test)
        
        # Confusion Matrix and Classification Report
        print('Decision Tree Results:')
        print(confusion_matrix(self.y_test, dtree_predictions))
        print(classification_report(self.y_test, dtree_predictions))

        print('Random Forest Results:')
        print(confusion_matrix(self.y_test, rfc_predictions))
        print(classification_report(self.y_test, rfc_predictions))

        # Visualize results
        df = pd.DataFrame(self.X_test.to_numpy(), columns=self.data.columns.drop('not.fully.paid'))
        df['dtree.prediction'] = dtree_predictions
        df['rfc.prediction'] = rfc_predictions
        print(df.head(20))
        sns.displot(data=df, x='fico', hue='dtree.prediction')
        sns.displot(data=df, x='fico', hue='rfc.prediction')
        plt.show()

if __name__ == '__main__':
    trees = Trees()
    trees.check_data()
    trees.eda()
    trees.clean()
    trees.train()
    trees.eval()
