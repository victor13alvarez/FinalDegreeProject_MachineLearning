import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Input Layer: It represent input image data. It will reshape image into single diminsion array. Example your image is 64x64 = 4096, it will convert to (4096,1) array.
# Conv Layer: This layer will extract features from image.
# Pooling Layer: This layerreduce the spatial volume of input image after convolution.
# Fully Connected Layer: It connect the network from a layer to another layer
# Output Layer: It is the predicted values layer.


# Constants of the program
FAST_RUN = True
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3
TRAIN_PATH = "train/"
TEST_PATH = "test/"
BATCH_SIZE = 50


def check_directory_files():
    files = os.listdir("./")
    if "model.h5" in files:
        return True
    else:
        return False


def prepare_data():
    filenames = os.listdir(TRAIN_PATH)
    categories = []
    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'dog':
            categories.append(1)
        else:
            categories.append(0)

    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })

    return df
    # print(df.head())
    # print(df.tail())
    # df['category'].value_counts().plot.bar()
    # plt.show()     #show number of dogs vs cats on dataset
    # see_sample_image(filenames)


def prepare_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))  # 2 because we have cat and dog classes

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model.summary()
    return model


def callbacks():
    earlystop = EarlyStopping(patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=2,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    callbacks = [earlystop, learning_rate_reduction]
    return callbacks


def split_data(df):
    df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})
    train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)

    # train_df['category'].value_counts().plot.bar()
    # plt.show()
    # validate_df['category'].value_counts().plot.bar()
    # plt.show()
    total_train = train_df.shape[0]
    total_validate = validate_df.shape[0]

    train_datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1. / 255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        TRAIN_PATH,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=BATCH_SIZE
    )

    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df,
        TRAIN_PATH,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=BATCH_SIZE
    )
    # check_generator_image(train_df, train_datagen)
    return train_generator, validation_generator, total_validate, total_train


def see_sample_image(f):
    sample = random.choice(f)
    image = load_img(TRAIN_PATH + sample)
    plt.imshow(image)
    plt.show()


def check_generator_image(train_df, train_datagen):
    example_df = train_df.sample(n=1).reset_index(drop=True)
    example_generator = train_datagen.flow_from_dataframe(
        example_df,
        TRAIN_PATH,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical'
    )
    plt.figure(figsize=(12, 12))
    for i in range(0, 15):
        plt.subplot(5, 3, i + 1)
        for X_batch, Y_batch in example_generator:
            image = X_batch[0]
            plt.imshow(image)
            break
    plt.tight_layout()
    plt.show()


def fit_model(model, train_generator, validation_generator, total_validate, total_train, callbacks):
    epochs = 3 if FAST_RUN else 50
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=total_validate // BATCH_SIZE,
        steps_per_epoch=total_train // BATCH_SIZE,
        callbacks=callbacks
    )
    model.save_weights("model.h5")
    return history, epochs


def visualize_training(history, epochs):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.plot(history.history['loss'], color='b', label="Training loss")
    ax1.plot(history.history['val_loss'], color='r', label="validation loss")
    ax1.set_xticks(np.arange(1, epochs, 1))
    ax1.set_yticks(np.arange(0, 1, 0.1))

    ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax2.plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
    ax2.set_xticks(np.arange(1, epochs, 1))

    legend = plt.legend(loc='best', shadow=True)
    plt.tight_layout()
    plt.show()


def prepare_test_data():
    test_filenames = os.listdir(TEST_PATH)
    test_df = pd.DataFrame({
        'filename': test_filenames
    })
    nb_samples = test_df.shape[0]
    test_gen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_gen.flow_from_dataframe(
        test_df,
        TEST_PATH,
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return test_generator, nb_samples, test_df


def predict(model, test_generator, nb_samples, test_df, train_generator):
    predict = model.predict(test_generator, steps=np.ceil(nb_samples / BATCH_SIZE))
    test_df['category'] = np.argmax(predict, axis=-1)
    label_map = dict((v, k) for k, v in train_generator.class_indices.items())
    test_df['category'] = test_df['category'].replace(label_map)
    test_df['category'] = test_df['category'].replace({'dog': 1, 'cat': 0})
    test_df['category'].value_counts().plot.bar()
    plt.show()
    return test_df


def show_image_result(test_df):
    sample_test = test_df.head(18)
    sample_test.head()
    plt.figure(figsize=(12, 24))
    for index, row in sample_test.iterrows():
        filename = row['filename']
        category = row['category']
        if category == 0:
            category = 'cat'
        else:
            category = 'dog'
        img = load_img(TEST_PATH + filename, target_size=IMAGE_SIZE)
        plt.subplot(6, 3, index + 1)
        plt.imshow(img)
        plt.xlabel(filename + '(' + "{}".format(category) + ')')
    plt.tight_layout()
    plt.show()


def load_trained_model(weights_path):
    model = prepare_model()
    model.load_weights(weights_path)
    return model


def main():
    df = prepare_data()
    mod = prepare_model()
    cb = callbacks()
    data = split_data(df)
    if not check_directory_files():
        data2 = fit_model(mod, data[0], data[1], data[2], data[3], cb)
        visualize_training(data2[0], data2[1])
    loadedmod = load_trained_model("model.h5")
    data3 = prepare_test_data()
    test_df = predict(loadedmod, data3[0], data3[1], data3[2], data[0])
    show_image_result(test_df)
