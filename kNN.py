#Klasyfikator k-NN

from data_loader import DataLoader
from data_preprocessing import Preprocessor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths


class k_NN:

    def __init__(self, path_to_image, path_to_dictionary):

        self.path_to_image = path_to_image
        self.path_to_dictionary = path_to_dictionary


    def kNN_model(self):
        input_directory = list(paths.list_images(self.path_to_dictionary))

        preproc = Preprocessor(224, 224)
        loader = DataLoader(preprocessor=[preproc])

        (images, labels) = loader.data_load_kNN(input_directory)
        images = images.reshape((images.shape[0], 224*224*3))

        labels_encoded = LabelEncoder()
        labels = labels_encoded.fit_transform(labels)

        class_mapping = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 4: 'sadness'}

        (trainX, testX, trainY, testY) = train_test_split(images, labels,
            test_size=0.25, random_state=55)

        model = KNeighborsClassifier(n_neighbors=15)
        model.fit(trainX, trainY)

        (image, label) = loader.data_load_kNN(self.path_to_image)
        image = image.reshape((image.shape[0], 224 * 224 * 3))

        prediction = model.predict(image)[0]
        predicted_label = class_mapping.get(prediction)

        return label, predicted_label
