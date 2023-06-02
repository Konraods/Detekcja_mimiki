#Klasyfikator k-NN
from GUI import MainWindow
from data_loader import DataLoader
from data_preprocessing import Preprocessor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths

#knn_path = "input"
knn_path = MainWindow.select_image
#Sciezka do zdjec
input_path = list(paths.list_images(knn_path))

preproc = Preprocessor(224, 224)
loader = DataLoader(preprocessor=[preproc])
(images, labels) = loader.data_load_kNN(input_path)
images = images.reshape((images.shape[0], 224*224*3))

labels_encoded = LabelEncoder()
labels = labels_encoded.fit_transform(labels)

class_mapping = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 4: 'sadness'}

(trainX, testX, trainY, testY) = train_test_split(images, labels,
    test_size=0.25, random_state=55)

model = KNeighborsClassifier(n_neighbors=15)
model.fit(trainX, trainY)

#print(classification_report(testY, model.predict(testX),
    #target_names=labels_encoded.classes_))
