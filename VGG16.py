#siec neuronowa VGG16 || Tworzenie modelu, wczytanie modelu, przewidywanie emocji na podstawie stworzonego modelu
import os
import shutil
import stat
import numpy as np
from imutils import paths
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import load_img, img_to_array
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense


from data_loader import DataLoader
from data_preprocessing import Preprocessor

class VGG_16:

    def __init__(self, path_to_dictionary, width=224, height=224):
        self.width = width
        self.height = height
        self.path_to_dictionary = path_to_dictionary
        self.preproc = Preprocessor(width=self.width, height=self.height)
        self.loader = DataLoader(preprocessor=[self.preproc])
        self.input_shape = (self.width, self.height, 3)


    def delete_content(self, redo_func, path, err):
        os.chmod(path, stat.S_IWRITE)
        redo_func(path)

    def delete_dictionaries(self):
        #ścieżka do folderów train i test
        try:
            shutil.rmtree(f"{os.path.dirname(self.path_to_dictionary)}" + "\\test", onerror=self.delete_content)
        except OSError:
            pass

        try:
            shutil.rmtree(f"{os.path.dirname(self.path_to_dictionary)}" + "\\train", onerror=self.delete_content)
        except OSError:
            pass

        try:
            shutil.rmtree("input_vgg16", onerror=self.delete_content)
        except OSError:
            pass

        try:
            os.makedirs("model")
        except OSError:
            pass

    def vgg16_model(self):

        input_path = list(paths.list_images(self.path_to_dictionary))

        # Pobiera liste podfolderow z katalogu
        folders = [f for f in os.listdir(input_path[0].split(os.path.sep)[-3]) if
                   os.path.isdir(f"{input_path[0].split(os.path.sep)[-3]}\\" + f)]

        self.loader.data_load_VGG(input_path=input_path, emotions=folders)

        train_data = "train"
        validation_data = "test"

        nb_train_samples = len(list(paths.list_images(train_data)))
        nb_validation_samples = len(list(paths.list_images(validation_data)))
        epochs = 15
        batch_size = 16

        model = Sequential()
        model.add(Conv2D(32, (2, 2), input_shape=self.input_shape))
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

        model.add(Dense(5))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
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
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
            validation_data,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='categorical')

        model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size)
        model_name = "model\\all_emotions.h5"
        model.save(model_name)


        # Usuwanie podfolderów z zawartością żeby móc stworzyć model dla nowych emocji
        shutil.rmtree(f"{os.path.dirname(self.path_to_dictionary)}" + "\\test", onerror=self.delete_content)
        shutil.rmtree(f"{os.path.dirname(self.path_to_dictionary)}" + "\\train", onerror=self.delete_content)

        return model_name


    def prediction(self, nnmodel, path_to_image):

        model = load_model(nnmodel[0])


        # Wczytanie i przetworzenie obrazu
        image = load_img(path_to_image[0], target_size=(self.width, self.height))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        class_mapping = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 4: 'sadness'}
        prediction = model.predict(image)

        # Pobierz indeks klasy z najwyższym prawdopodobieństwem
        predicted_emotion = np.argmax(prediction)

        # Przypisz emocję na podstawie indeksu
        predicted_emotion = class_mapping[predicted_emotion]

        return predicted_emotion
