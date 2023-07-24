'''
Linear Regression

Linear regression is a common statistical method used to predict the relationship between two variable.
It assumes a linear relationship and attempts to find a line of best fit by minimizing the distance 
between all the points and their distance to the line. The most common approach is to use a least
squares method.

This script will use linear regression on an ecommerce dataset from a clothing company to determine 
if they should focus their development effors on a mobile app or website to generate sales.
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

class LinearRegressor:
    def __init__(self) -> None:
        self.data = pd.read_csv('./data/ecommerce_data.csv')
        self.model = None
        self.X_test = None
        self.y_test = None
    
    def check_data(self):
        """
        Key features:
        1. Avg Session Length
        2. Time on App
        3. Time on Website
        4. Length of Membership

        """
        print(f'{self.data.head(5)}\n\n')  # print first 5 elements
        print(f'{self.data.describe()}\n\n')  # check stats
        print(f'{self.data.info()}\n\n')  # check features

    def eda(self):
        """
        Exploratory Data Analysis

        """
        sns.jointplot(x=self.data['Time on App'], y=self.data['Yearly Amount Spent'])
        sns.jointplot(x=self.data['Time on Website'], y=self.data['Yearly Amount Spent'])
        sns.jointplot(x=self.data['Time on Website'], y=self.data['Yearly Amount Spent'], kind='hex')
        sns.pairplot(self.data)

        # The two variables that had the highest correlation as seen on the pairplot
        sns.lmplot(data=self.data, x='Length of Membership', y='Yearly Amount Spent')

        plt.show()
    
    def train(self):
        """
        Train the model

        """
        # Split data into features and labels
        X = self.data[['Avg. Session Length', 'Time on App','Time on Website', 
                       'Length of Membership']]
        y = self.data[['Yearly Amount Spent']]  # the label we want to predict

        # Split data into training and testing subsets
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.3, 
                                                                      random_state=101)

        # Train the model
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        # Model metrics
        print(f'Linear Regression Coefficients: {self.model.coef_}')  # coefficients for each feature
        print(f'Linear Regression Intercept: {self.model.intercept_}')  # intercept

    def eval(self):
        """
        Evaluate the model

        The model shows a good correlation between our actual values and predicted values.
        There is still some noise/error, but overall we are explaing ~99% of the variance
        """
        # Scatter plot
        predictions = self.model.predict(self.X_test)
        plt.scatter(self.y_test, predictions)
        plt.xlabel('Actual values')
        plt.ylabel('Predicited Values')

        # Residual plot
        sns.displot((self.y_test-predictions))
        plt.show()

        # Calculate Error
        print('MAE: {}'.format(metrics.mean_absolute_error(self.y_test, predictions)))
        print('MSE: {}'.format(metrics.mean_squared_error(self.y_test, predictions)))
        print('RMSE: {}'.format(np.sqrt(metrics.mean_squared_error(self.y_test, predictions))))
        print('Residual: {}'.format(metrics.explained_variance_score(self.y_test, predictions)))  # how much variance the model explained
        print('\n')

    def interpret(self):
        """
        Interpret the data

        Given we have an accurate model, how should we interpret the data? If we analyze the 
        coefficients of the linear fit, we can make the following conclusions:

        1. Every unit of increase in session length corresponds to an increase of $26 
        2. Every unit of increase in time on app corresponds to an increase of $39 
        3. Every unit of increase in time on website corresponds to an increase of $0.19 
        4. Every unit of increase in membership corresponds to an increase of $61 

        So depending on this business, we could advise this company to either focus on app 
        development, improving the website to catch up to gains made on the app, 
        or focus on increasing customer membership time

        """
        cdf = pd.DataFrame(self.model.coef_.reshape(-1,1), self.X_test.columns, columns=['Coeff'])
        cdf.plot.bar()
        plt.show()
        print(cdf)

if __name__ == '__main__':
    lr = LinearRegressor()
    lr.check_data()
    lr.eda()
    lr.train()
    lr.eval()
    lr.interpret()
