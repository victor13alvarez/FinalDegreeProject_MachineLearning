import keras
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


# the data, split between train and test sets

# CONSTANTS

def start_program():
    split_data()


def split_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape, y_train.shape)

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes=None)
    y_test = to_categorical(y_test, num_classes=None)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    create_model(input_shape, y_train, y_test, x_train, x_test)


def create_model(input_shape, y_train, y_test, x_train, x_test):
    batch_size = 128
    num_classes = 10
    epochs = 50
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='Adam',
                  metrics=['accuracy'])
    train_model(model, y_train, y_test, x_train, x_test, batch_size, epochs)


def train_model(model, y_train, y_test, x_train, x_test, batch_size, epochs):
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                     validation_data=(x_test, y_test))
    print("The model has successfully trained")
    model.save('mnist.h5')
    print("Saving the model as mnist.h5")
    evaluate_model(model, x_test, y_test)


def evaluate_model(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('THIS IS A CN MODEL TRAINING \n')
    start_program()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
