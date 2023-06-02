#Klasyfikator k-NN

from data_loader import DataLoader
from data_preprocessing import Preprocessor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths


class k_NN:
    #Konstruktor ze sciezka do folderow ze zdjeciami oraz sciezka do badanego zdjecia
    def __init__(self, path_to_image, path_to_dictionary):
        self.path_to_image = path_to_image
        self.path_to_dictionary = path_to_dictionary
        self.preproc = Preprocessor(224, 224)
        self.loader = DataLoader(preprocessor=[self.preproc])

    def kNN_model(self):

        (images, labels) = self.loader.data_load_kNN(list(paths.list_images(self.path_to_dictionary)))
        images = images.reshape((images.shape[0], 224*224*3))

        labels_encoded = LabelEncoder()
        labels = labels_encoded.fit_transform(labels)

        (trainX, testX, trainY, testY) = train_test_split(images, labels,
            test_size=0.25, random_state=55)

        model = KNeighborsClassifier(n_neighbors=15)
        model.fit(trainX, trainY)

        return model

    def kNN_prediction(self, model):

        (image, label) = self.loader.data_load_kNN(self.path_to_image)
        image = image.reshape((image.shape[0], 224 * 224 * 3))

        class_mapping = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 4: 'sadness'}

        prediction = model.predict(image)[0]
        predicted_label = class_mapping.get(prediction)

        return label, predicted_label
