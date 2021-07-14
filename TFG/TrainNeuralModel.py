import PrepareDataForNeuralModel as Data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.vis_utils import plot_model
import os
import pandas as pd
import numpy

FolderResultsPath = "Results"
FinalCSV = "_FinalCSV.csv"
FinalAUXCSV = "_AuxCSV.csv"


def normalize(df):
    result = df.copy()
    min_duration = 0
    max_duration = 0
    mean_duration = 0
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        mean_value = df[feature_name].mean()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        if feature_name == "DURATION":
            min_duration = min_value
            max_duration = max_value
            mean_duration = mean_value
    return result, min_duration, max_duration, mean_duration


def undo_normalize(df, max_dur, min_dur):
    result = df.copy()
    for feature_name in df:
        result[feature_name] = df[feature_name] * (max_dur - min_dur) + min_dur
    return result


def trainModel():
    # Hyperparameters
    learning_rate = 0.001
    n_epochs = 1000
    batch_size = 10
    validation_split = 0.1
    activation = 'relu'

    df = Data.prepareDataSetForNeuralModel()
    # print(df.shape)
    # print(df.head)

    ### Normalize values between 0 and 1
    df_normalized, minDuration, maxDuration, meanDuration = normalize(df)

    # print(df_normalized.shape)
    # print(df_normalized.head)

    # Regression DataModel (Regression refers to predictive modeling problems that involve predicting a numeric value given an input.)

    # split into input (X) and output (y) variables
    # split into input (X) and output (y) variables
    x = df_normalized.iloc[:, :-1].values
    y = df_normalized.iloc[:, -1].values  # 1 output
    n_features = x.shape[1]  # 20 valores de entrada
    ##LSTM
    # n_features = x.reshape((len(x), 1, 1))
    ##FINLSTM
    # split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
    # define the keras model #sigmoid
    model = Sequential()
    # input_dim=n_features
    # , activation='relu'
    model.add(Dense(32, input_dim=n_features, activation='relu'))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # compile the keras model
    # Alternative
    # model.compile(loss='mse', optimizer='adam')
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error', 'mean_absolute_error',
                                                                        'mean_absolute_percentage_error',
                                                                        'cosine_proximity'])
    model.summary()
    plot_model(model, to_file=os.path.join(FolderResultsPath, 'ModelSummary.png'), show_shapes=True)

    # fit the keras model on the dataset
    # verbose 0, 1 or 2 you just say how do you want to 'see' the training progress for each epoch.
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=150, batch_size=32, verbose=0, shuffle=True)

    # LSTM
    # history = model.fit(x_train, y_train, validation_split=0.2, epochs=150, batch_size=32, verbose=0, shuffle=True)
    # LSTM

    # Evalaute Model
    predictions = model.predict(x_test)
    error = mean_absolute_error(y_test, predictions)
    score = model.evaluate(x_test, y_test, verbose=0)

    # Plot training history
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(os.path.join('Results', 'TrainingHistory.png'))
    plt.show()

    # Save results
    plt.plot(history.history['mean_squared_error'])
    plt.title('Mean Squared Error')
    plt.ylabel('mean')
    plt.xlabel('epoch')
    plt.savefig(os.path.join('Results', 'mean_squared_error.png'))
    plt.show()

    plt.plot(history.history['mean_absolute_error'])
    plt.title('Mean Absolute Error')
    plt.ylabel('mean')
    plt.xlabel('epoch')
    plt.savefig(os.path.join('Results', 'mean_absolute_error.png'))
    plt.show()

    plt.plot(history.history['mean_absolute_percentage_error'])
    plt.title('Mean Absolute Percentage Error')
    plt.ylabel('mean')
    plt.xlabel('epoch')
    plt.savefig(os.path.join('Results', 'mean_absolute_percentage_error.png'))
    plt.show()

    plt.plot(history.history['cosine_proximity'])
    plt.title('Cosine Proximity')
    plt.ylabel('cosine proximity')
    plt.xlabel('epoch')
    plt.savefig(os.path.join('Results', 'cosine_proximity.png'))
    plt.show()

    # SAVE RESULTS
    predictions = numpy.array(predictions)
    predictionsToList = []
    for i in predictions:
        for j in i:
            predictionsToList.append(j)

    final_df = pd.DataFrame([y_test, predictionsToList])
    final_df = final_df.transpose()
    final_df.columns = ['Real duration', 'Prediction duration']
    final_df = undo_normalize(final_df, maxDuration, minDuration)
    final_df.to_csv(FolderResultsPath + '/' + FinalCSV, index=False)
    model.save(FolderResultsPath + '/' + 'mnist.h5')
    aux_df = pd.DataFrame([minDuration, maxDuration, meanDuration, error, score[0], score[1]])
    aux_df = aux_df.transpose()
    aux_df.columns = ['Min Duration', 'Max Duration', "Mean Duration", "Error", "loss", "val_loss"]
    aux_df.to_csv(FolderResultsPath + '/' + FinalAUXCSV, index=False)
