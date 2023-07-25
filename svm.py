'''
Support Vector Machines (SVM)

SVMs are supervised learning models that can be used for classification and regression analysis.
This algorithm is useful for data that can be clearly seperated into distint categories.

This script will demonstrate how to use a SVM on the famous Iris flower dataset. It will also 
demonstrate how to use GridSearchCV to find optimal parameters.
'''

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

class SVM:
    def __init__(self) -> None:
        self.data = sns.load_dataset('iris')
        self.model = None
        self.optimized_model = None
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

        We will create a single pairplot to see what flower species are the most separable
        """
        sns.pairplot(self.data, hue='species', palette='turbo')
        plt.show()

    def train(self):
        """
        Train the model

        """
        # Split data into features and labels
        X = self.data.drop('species', axis=1)
        y = self.data['species']  # the label we want to predict

        # Split data into training and testing subsets
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.3, 
                                                                      random_state=101)
        
        # Train
        self.model = SVC()
        self.model.fit(X_train, y_train)

        # We can use gridsearch to find the optimial parameter for the SVC and compare models
        param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 
                      'kernel': ['rbf']}  # parameters to optimize
        self.optimized_model = GridSearchCV(SVC(), param_grid=param_grid, verbose=3)
        self.optimized_model.fit(X_train, y_train)
        print('Optimial Parameters: {}'.format(self.optimized_model.best_params_))

    def eval(self):
        """
        Evaluate the model

        Compare the standard SVM to the optimized SVM. Since our original model was already very
        accurate, we do not expect much improvemtn from using gridsearch here.

        """
        predictions = self.model.predict(self.X_test)
        grid_predictions = self.optimized_model.predict(self.X_test)
        
        # Confusion Matrix and Classification Report
        print('Standard Model Results:')
        print(confusion_matrix(self.y_test, predictions))
        print(classification_report(self.y_test, predictions))

        print('Optimized Model Results:')
        print(confusion_matrix(self.y_test, grid_predictions))
        print(classification_report(self.y_test, grid_predictions))

        # Visualize - Actual vs Predicted for two features
        all_predictions = self.model.predict(self.data.drop('species', axis=1))
        df = self.data.copy()
        df['predicted'] = all_predictions
        print(df.head())
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species', ax=axs[0])
        sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='predicted', ax=axs[1])
        axs[0].set_title('Actual')
        axs[1].set_title('Predicted')
        plt.show()

if __name__ == '__main__':
    svm = SVM()
    svm.check_data()
    svm.eda()
    svm.train()
    svm.eval()
