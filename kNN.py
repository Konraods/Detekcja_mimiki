#Klasyfikator k-NN || Tworzenie modelu, przewidywanie emocji na podstawie stworzonego modelu
from data_loader import DataLoader
from data_preprocessing import Preprocessor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths

class k_NN:
    """Konstruktor ze sciezka do folderow ze zdjeciami
    oraz preprocesorem i dataloaderem do modyfikacji zdjec"""
    def __init__(self, path_to_dictionary):
        self.path_to_dictionary = path_to_dictionary
        self.preproc = Preprocessor(224, 224)
        self.loader = DataLoader(preprocessor=[self.preproc])

    """Metoda do tworzenia modelu kNN"""
    def kNN_model(self):
        #Przeksztalcanie zdjec na jednowymiarowe szeregi dla klasyfikatora kNN, tworzenie listy etykiet zdjec
        (images, labels) = self.loader.data_load_kNN(input_path=list(paths.list_images(self.path_to_dictionary)))
        images = images.reshape((images.shape[0], 224*224*3))

        #Zmiana etykiet z lancuchow znakow na cyfry
        labels_encoded = LabelEncoder()
        labels = labels_encoded.fit_transform(labels)

        #Podzielenie zdjec na zbiory treningowy i testowy w stosunku 4:1
        (trainX, testX, trainY, testY) = train_test_split(images, labels,
            test_size=0.25, random_state=55)

        #Stworzenie modelu
        model = KNeighborsClassifier(n_neighbors=15)
        model.fit(X=trainX, y=trainY)

        return model

    """Metoda do predykcji emocji z wybranego zdjecia"""
    def kNN_prediction(self, model, path_to_image):
        # Przeksztalcanie zdjecia na jednowymiarowe szereg dla klasyfikatora kNN, tworzenie etykiety zdjecia
        (image, label) = self.loader.data_load_kNN(input_path=path_to_image)
        image = image.reshape((image.shape[0], 224 * 224 * 3))

        #Slownik do odszyfrowania etykiety
        class_mapping = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 4: 'sadness'}

        #Detekcja mimiki na podstawie modelu
        prediction = model.predict(image)[0]
        #Odszyfrowanie etykiety na podstawie slownika
        predicted_label = class_mapping.get(prediction)

        return label, predicted_label
