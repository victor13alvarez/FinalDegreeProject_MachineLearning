import TrainNeuralModel as Train
import os
import tensorflow.keras.models as ModelImport
import pandas as pd

FinalCSV = "_FinalCSV.csv"
FinalAUXCSV = "Results/_AuxCSV.csv"
testArray1 = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.8, 0.7, 0.9, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]
testArray2 = [[0.5, 0.3, 0.4, 0.6, 0.5, 0.5, 0.6, 0.4, 0.5, 0.3, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]]
testArray3 = [[0.2, 0.3, 0.4, 0.9, 0.8, 0.6, 0.7, 0.2, 0.1, 0.9, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1]]
testArray4 = [[0.4, 0.5, 0.2, 0.2, 0.9, 0.9, 0.7, 0.7, 0.2, 0.2, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1]]

if __name__ == '__main__':
    Train.trainModel()
    if "mnist.h5" not in os.listdir("./Results"):
        Train.trainModel()

    model = ModelImport.load_model("./Results/mnist.h5")
    # input_data = receiveInput()
    print("The estimated duration of match is ", model.predict(testArray1), "seconds")
    print("The estimated duration of match is ", model.predict(testArray2), "seconds")
    print("The estimated duration of match is ", model.predict(testArray3), "seconds")
    print("The estimated duration of match is ", model.predict(testArray4), "seconds")
    df = pd.read_csv(FinalAUXCSV)
    print(df)
