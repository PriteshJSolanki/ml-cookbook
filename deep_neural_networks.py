'''
Deep Neural Networks

This script will demonstrate how to build and train a multi-layer feedforward neural network
(i.e deep neural network). The self.model will train off of a dataset from LendingClub taken from Kaggle.
The goal of the self.model will be to predict whether or not a customer will be able to pay off a loan.

'''
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

class DNN:
    def __init__(self) -> None:
        self.data_info = pd.read_csv('./data/lending_club_info.csv', index_col='LoanStatNew')
        self.data = pd.read_csv('./data/lending_club_loans.csv')
        self.model = None
        self.scaler = None
        self.X_test = None
        self.y_test = None
    
    def feat_info(self, col_name):
        print(self.data_info.loc[col_name]['Description'])

    def fill_mort_acc(self, total_acc, mort_acc, total_acc_avg):
        if np.isnan(mort_acc):
            return total_acc_avg[total_acc]
        else:
            return mort_acc

    def check_data(self):
        """
        Check data

        """
        self.feat_info('loan_amnt')
        print(self.data.head())
        print(self.data.info())
        print(self.data.describe())

    def eda(self):
        """
        Exploratory Data Analysis

        """
        sns.countplot(self.data, x='loan_status')  # data is slightly imbalanced
        plt.show()
        sns.displot(self.data, x='loan_amnt', kde=True, aspect=2, height=5)
        plt.show()

        # Correlation
        plt.figure(figsize=(10,6))
        sns.heatmap(self.data.corr(numeric_only=True), annot=True, cmap='viridis')
        plt.show()

        # Installment and Loan Amount are almost perfectly correlated
        sns.scatterplot(self.data, x='installment', y='loan_amnt')
        plt.show()

        # Check relationship between loan amount and status
        # There is a slight relationship when the loan is higher and not being paid off
        sns.boxplot(self.data, x='loan_status', y='loan_amnt')  
        self.data.groupby('loan_status')['loan_amnt'].describe()
        plt.show()

        # Grade/Subgrade
        self.data['grade'].unique()
        self.data['sub_grade'].unique()
        sns.countplot(self.data, x='grade', hue='loan_status')
        plt.show()
        plt.figure(figsize=(10,6))
        subgrade_order = sorted(self.data['sub_grade'].unique())
        sns.countplot(self.data, x='sub_grade', hue='loan_status', order=subgrade_order, palette='coolwarm')
        plt.show()

        # Look only at F-G loans
        f_and_g = self.data[(self.data['grade']=='G') | (self.data['grade']=='F')]
        plt.figure(figsize=(10,6))
        subgrade_order = sorted(f_and_g['sub_grade'].unique())
        sns.countplot(f_and_g, x='sub_grade', hue='loan_status', order=subgrade_order, palette='coolwarm')
        plt.show()

    def preprocess(self):
        """
        Preprocess the data before training

        """
        
        ####################################################################################################
        # Missing Data
        ####################################################################################################
        # How much missing data do we have?
        missing_pct = self.data.isnull().sum().sort_values(ascending=False) / len(self.data) * 100
        print(f'Missing Data %: {missing_pct}')  # get missing data as a %

        # 5% of missing data is from emp_title
        print(self.data['emp_title'].nunique())
        print(self.data['emp_title'].value_counts())
        self.data = self.data.drop('emp_title', axis=1) # too many titles to edit, drop them

        # emp_length & title are the next largest missing values
        print(sorted(self.data['emp_length'].dropna().unique()))
        emp_len_order = ['< 1 year',
                        '1 year',
                        '2 years',
                        '3 years',
                        '4 years',
                        '5 years',
                        '6 years',
                        '7 years',
                        '8 years',
                        '9 years',
                        '10+ years']
        plt.figure(figsize=(10,4))
        sns.countplot(self.data, x='emp_length', order=emp_len_order, hue='loan_status', palette='coolwarm')

        # # Whats the ratio of paid to unpaid per employment length?
        emp_unpaid = self.data[self.data['loan_status']=='Charged Off'].groupby('emp_length').count()['loan_status']
        emp_paid = self.data[self.data['loan_status']=='Fully Paid'].groupby('emp_length').count()['loan_status']
        ratio = (emp_unpaid/emp_paid)
        ratio.info()
        ratio.plot(kind='bar')
        self.data.drop('emp_length', axis=1, inplace=True)

        # purpose & title
        print(self.feat_info('purpose'))
        self.data = self.data.drop('title', axis=1)  # title and purpose are the same

        # mort_acc
        # the missing mort_acc accounts for nearly 10% of our data
        print(self.data.isnull().sum())
        print(self.data['mort_acc'].value_counts())
        # which feature correlate to mortage accounts?
        print(self.data.corr(numeric_only=True)['mort_acc'].sort_values(ascending=False))
        # reasonable correlation between total_acc and mort_acc. Get average of mort_accs 
        # based on total accounts
        total_acc_avg = self.data.groupby('total_acc').mean(numeric_only=True)['mort_acc']     
        self.data['mort_acc'] = self.data.apply(lambda x: self.fill_mort_acc(x['total_acc'],x['mort_acc'], total_acc_avg), axis=1)

        # revol_util & pub_rec_bankruptcies
        # drop both since they are so few
        self.data.drop('revol_util', axis=1, inplace=True)
        self.data.drop('pub_rec_bankruptcies', axis=1, inplace=True)

        # Create loan_repaid status
        self.data['loan_repaid'] = self.data['loan_status'].map({'Fully Paid':1, 'Charged Off':0})
        self.data[['loan_repaid', 'loan_status']]

        # loan status is a duplicate of loan_repaid
        self.data.drop('loan_status', axis=1, inplace=True)

        print(self.data.isnull().sum())  # print final missing values

        ####################################################################################################
        # Configure Categorical Data
        ####################################################################################################

        # List non-numerical data
        print(self.data.select_dtypes(['object']).columns)

        # term
        # self.self.feat_info('term')
        print(self.data['term'].value_counts())  # there is a numeric relationship between 36-mo and 60-mo
        self.data['term'] = self.data['term'].apply(lambda term: int(term[:3]))

        # grade
        self.data.drop('grade', axis=1, inplace=True)  # grade is already in sub_grade

        # subgrade
        # convert to dummy variables
        dummies = pd.get_dummies(self.data['sub_grade'], drop_first=True)
        self.data = pd.concat([self.data.drop('sub_grade',axis=1),dummies], axis=1)

        # remaining categories
        dummies = pd.get_dummies(self.data[['verification_status', 'application_type','initial_list_status','purpose' ]],
                                drop_first=True)
        self.data = self.data.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
        self.data = pd.concat([self.data,dummies], axis=1)

        # home ownership
        print(self.data['home_ownership'].value_counts())
        self.data['home_ownership'].replace(['NONE', 'ANY'], 'OTHER', inplace=True)  # get rid of NONE and ANY
        dummies = pd.get_dummies(self.data['home_ownership'], drop_first=True)
        self.data = pd.concat([self.data.drop('home_ownership',axis=1),dummies], axis=1)

        # address
        self.data['zip_code'] = self.data['address'].apply(lambda address: address[-5:])  # extract only zipcode
        dummies = pd.get_dummies(self.data['zip_code'], drop_first=True)
        self.data = pd.concat([self.data.drop('zip_code',axis=1),dummies], axis=1)
        self.data.drop('address', axis=1, inplace=True)

        # issue_d
        self.data.drop('issue_d',axis=1, inplace=True)  # could be data leakage, drop it

        # earliest_cr_line
        self.data['earliest_cr_line'] = self.data['earliest_cr_line'].apply(lambda date: int(date[-4:]))

    def train(self):
        """
        Train the model

        """
        # Split data
        X = self.data.drop('loan_repaid', axis=1)
        y = self.data['loan_repaid']
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=101)

        # Normalize
        # We need this step in order to correctly optimize the weights of the NN. The scaler will
        # fit the data from 0 to 1 and scale it.
        self.scaler = MinMaxScaler()
        X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # Train the NN
        self.model = Sequential()

        # input layer
        self.model.add(Dense(78,  activation='relu'))
        self.model.add(Dropout(0.2))

        # hidden layer
        self.model.add(Dense(39, activation='relu'))
        self.model.add(Dropout(0.2))

        # hidden layer
        self.model.add(Dense(19, activation='relu'))
        self.model.add(Dropout(0.2))

        # output layer
        self.model.add(Dense(units=1,activation='sigmoid'))

        # Compile
        self.model.compile(loss='binary_crossentropy', optimizer='adam')

        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)  

        self.model.fit(x=X_train, y=y_train, epochs=25, batch_size=256, validation_data=(self.X_test, self.y_test), 
                  callbacks=early_stop)
        
    def eval(self):
        """
        Evaluate the model

        """
        # model.save('project_model.h5')
        loss = pd.DataFrame(self.model.history.history) # type: ignore
        plt.cla()
        loss.plot()
        plt.show()

        predictions = (self.model.predict(self.X_test) > 0.5).astype("int32")
        print(classification_report(self.y_test, predictions))
        print(confusion_matrix(self.y_test, predictions))

        # Test with a random customer
        random.seed(101)
        random_id = random.randint(0, len(self.data))
        new_customer = self.data.drop('loan_repaid', axis=1).iloc[random_id]
        new_customer = self.scaler.transform(new_customer.values.reshape(1,76))  # type: ignore

        prediction = (self.model.predict(new_customer) > 0.5).astype("int32")
        actual = self.data['loan_repaid'].iloc[random_id]
        print('self.model Predicted: {}\nActual Result: {}'.format(prediction, actual))

if __name__ == '__main__':
    dnn = DNN()
    # dnn.check_data()
    # dnn.eda()
    dnn.preprocess()
    dnn.train()
    dnn.eval()
