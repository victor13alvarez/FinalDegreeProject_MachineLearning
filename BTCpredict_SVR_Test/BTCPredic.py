import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import ModelNeededData as MND

mnd = MND.DataStructure(30,[],'')

def setup_dataset():
    mnd.df = pd.read_csv("bitcoin.csv")
    show_data(mnd.df.head())
    modify_dataset()


def modify_dataset():
    mnd.df.drop(['Date'], 1, inplace=True)
    show_data(mnd.df.head())
    setup_prediction()


def setup_prediction():
    mnd.df['Prediction'] = mnd.df[['Price']].shift(-mnd.prediction_days)        # Create another column shifted 'n'  units up
    show_data(mnd.df.head())
    show_data(mnd.df.tail(30))
    x = np.array(mnd.df.drop(['Price'], 1))
    x = x[:len(mnd.df) - mnd.prediction_days]       # Drop all price data into numpy array and remove prediction data from the bottom

    y = np.array(mnd.df['Prediction'])  # Create the dependent data set # convert the data frame into a numpy array # Get all the values except last 'n' rows
    y = y[:-mnd.prediction_days]

    # Split the data into 80% training and 20% testing
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
    # set the predictionDays array equal to last 30 rows from the original data set
    mnd.predictiondays_array = np.array(mnd.df.drop(['Prediction'], 1))[-mnd.prediction_days:]
    print(mnd.predictiondays_array)
    setup_MLmodel(xtrain,xtest,ytrain,ytest)

def setup_MLmodel(xtrain,xtest,ytrain,ytest):
    # Create and Train the Support Vector Machine (Regression) using radial basis function
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.00001)
    svr_rbf.fit(xtrain, ytrain)
    test_MLmodel(svr_rbf,xtest,ytest)


def test_MLmodel(svr_rbf,xtest,ytest):
    svr_rbf_confidence = svr_rbf.score(xtest, ytest)
    print('SVR_RBF accuracy :', svr_rbf_confidence)
    # print the predicted values
    svm_prediction = svr_rbf.predict(xtest)
    print(svm_prediction)
    print()
    print(ytest)
    check_MLmodel(svr_rbf)


def check_MLmodel(svr_rbf ):
    # Print the model predictions for the next 30 days
    svm_prediction = svr_rbf.predict(mnd.predictiondays_array)
    print(svm_prediction)
    print()
    # Print the actual price for bitcoin for last 30 days
    print(mnd.df.tail(30))
    mnd.df.iloc[-30:]['Prediction'] = svm_prediction
    print(mnd.df.tail(30))


def show_data(data):
    print('Current DataSet \n')
    print(data, '\n')
