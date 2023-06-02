import os
import shutil
import stat
import cv2
from imutils import paths
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from data_loader import DataLoader
from data_preprocessing import Preprocessor


def redo_with_write(redo_func, path, err):
    os.chmod(path, stat.S_IWRITE)
    redo_func(path)

#ścieżka do folderów train i test
try:
    shutil.rmtree("path\\to\\test", onerror=redo_with_write)
    shutil.rmtree("path\\to\\train", onerror=redo_with_write)
except OSError:
    pass

try:
    os.makedirs("models")
except OSError:
    pass

width, height = 224, 224
input_shape = (width, height, 3)

# W folderze input znajduja sie podfoldery z emocjami (klasy)
input_path = list(paths.list_images("input"))



# Pobieramy listę podfolderów z katalogu "input"
folders = [f for f in os.listdir(input_path[0].split(os.path.sep)[-3]) if os.path.isdir(f"{input_path[0].split(os.path.sep)[-3]}\\" + f)]
#Lista na pary emocji
emotion_pairs = []

for idx, first in enumerate(folders):
    for second in folders[idx + 1:]:
        emotion_pairs.append((first, second))



preproc = Preprocessor(width, height)
loader = DataLoader(preprocessor=[preproc])

for pair in emotion_pairs:
    loader.data_load_VGG(input_path, emotions=pair)

    train_data = "train"
    validation_data = "test"

    nb_train_samples = len(list(paths.list_images("train")))
    nb_validation_samples = train = len(list(paths.list_images("test")))
    epochs = 15
    batch_size = 16

    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode='binary')

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    model.save(f'models\\{pair[0]}_{pair[1]}')


    def redo_with_write(redo_func, path, err):
        os.chmod(path, stat.S_IWRITE)
        redo_func(path)
    # Usuwanie podfolderów z zawartością żeby móc stworzyć model dla nowych emocji
    shutil.rmtree("path\\to\\test", onerror=redo_with_write)
    shutil.rmtree("path\\to\\train", onerror=redo_with_write)
