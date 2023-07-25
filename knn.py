'''
K-Nearest Neighbors

K Nearest Neighbors is a classification algorithm that is useful for categorizing data with limited
number of features and does not require assumptions about the meaning or shape of the data 
(non-parametric).

The algorithm works as such:
1. Calculate the distance from x to all other points in the dataset
2. Sorts the data by increasing distance to x
3. Predicts the label by using "k" closest points

Pros:
1. Simple
2. No training required
3. Good for multi-class problems

Cons:
1. High prediction cost for large datasets
2. Not good for data with large number of features
3. Categorical features don't work well

This script will use KNN on an anonymized dataset to predicts a class for a 
new data point based off of the features.
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

class KNN:
    def __init__(self) -> None:
        self.data = pd.read_csv('./data/anonymized_data.csv', index_col=0)  # use first col as index
        self.scaled_data = None
        self.model = None
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

        Since data is anonymized, we can just look at all of it in a pairplot
        """
        sns.pairplot(data=self.data, hue='TARGET CLASS')
        plt.show()
    
    def clean(self):
        """
        This step will standardize the variables to the same scale.

        Since KNN uses distance to other points to predict class, the scale of the variables matters.
        If some variables are on a larger scale, the prediction will be incorrect.

        """
        scaler = StandardScaler()
        scaler.fit(self.data.drop('TARGET CLASS', axis=1))  # only scale features
        self.scaled_data = scaler.transform(self.data.drop('TARGET CLASS', axis=1))  # Apply the scaling

        # Convert scaled features to a dataframe and check results
        df = pd.DataFrame(data=self.scaled_data, columns=self.data.columns[:-1])
        print(df.head())  # check to ensure it worked as expected


    def train(self, k:int=None):
        """
        Train the model

        Args:
            k (optional): k-value for KNN algorithm. Defaults to None. 
                          Attempts to find the value if None.

        """
        # Split data into features and labels
        X = self.scaled_data
        y = self.data['TARGET CLASS']  # the label we want to predict

        # Split data into training and testing subsets
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.3, 
                                                                      random_state=101)

        # KNN
        # Find the best k value using the elbow method
        if k is None:
            err_rate = []
            for i in range(1, 40):
                self.model = KNeighborsClassifier(n_neighbors=i)
                self.model.fit(X_train, y_train)
                pred_i = self.model.predict(self.X_test)
                err_rate.append(np.mean(pred_i != self.y_test))
            
            # plot results
            plt.figure(figsize=(10,6))
            plt.plot(range(1,40), err_rate, color='blue', linestyle='dashed', marker='o',
                    markerfacecolor='red', markersize=10)
            plt.title('Error Rate vs. K Value')
            plt.xlabel('K')
            plt.ylabel('Error Rate')
            plt.show()
        
        else:
            self.model = KNeighborsClassifier(n_neighbors=k)
            self.model.fit(X_train, y_train)


    def eval(self):
        """
        Evaluate the model

        """
        predictions = self.model.predict(self.X_test)
        
        # Create a df with predictions and actual target clases in order to visualize accuracy
        df = pd.DataFrame(self.X_test, columns=self.data.columns[:-1])
        df['PREDICTED CLASS'] = predictions
        df['ACTUAL CLASS'] = self.y_test.to_numpy()
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        sns.scatterplot(data=df, x='WTT', y='PTI', hue='ACTUAL CLASS', ax=axs[0])
        sns.scatterplot(data=df, x='WTT', y='PTI', hue='PREDICTED CLASS', ax=axs[1])
        axs[0].set_title('Actual')
        axs[1].set_title('Predicted')
        plt.show()

        # Confusion Matrix and Classification Report
        print(confusion_matrix(self.y_test, predictions))
        print(classification_report(self.y_test, predictions))

    def test(self):
        """
        Test the model with a random dataset

        """
        # Construct a sample dataframe
        df = pd.DataFrame([[-0.123542, 0.185907,-0.913431,0.319629,-1.033637,-2.308375,-0.798951,-1.482368,-0.949719,-0.643314]], 
                          columns=self.data.columns[:-1])
        predictions = self.model.predict(df)
        
        # Print
        for sample_num, pred in enumerate(predictions):
            print(f'{sample_num}: Target Class = {pred}')

if __name__ == '__main__':
    knn = KNN()
    knn.check_data()
    knn.eda()
    knn.clean()
    knn.train(k=30)
    knn.eval()
    knn.test()
