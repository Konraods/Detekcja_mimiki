#Aplikacja okienkowa do obslugi klasyfikatora kNN i sieci neuronowej VGG16
import sys
#import VGG16
import os
import easygui as eg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout
from kNN import k_NN

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Ustawienia okna
        self.setGeometry(150, 150, 450, 150)
        self.setWindowTitle('Aplikacja do detekcji mimiki twarzy')

        # Przycisk - zdjecia
        self.button = QPushButton('Pick image', self)
        self.button.clicked.connect(self.select_image)

        # Przycisk - model
        self.button1 = QPushButton('Create models', self)
        self.button1.setEnabled(False)
        self.button1.clicked.connect(self.create_models)

        # Przycisk - k-nn
        self.button2 = QPushButton('k-NN classificator', self)
        self.button2.setEnabled(False)
        self.button2.clicked.connect(self.kNN_predict)

        # Przycisk - vgg16
        self.button3 = QPushButton('VGG16 neural network', self)
        self.button3.setEnabled(False)
        #self.button3.clicked.connect(self.on_button_click)

        # Etykieta - klasa
        self.label = QLabel('Labeled emotion: ', self)
        self.label.setAlignment(Qt.AlignCenter)

        # Etykieta - przewidywania
        self.label1 = QLabel('Predicted emotion:', self)
        self.label1.setAlignment(Qt.AlignCenter)

        # Uklad pionowy dla przyciskow
        vbox1 = QVBoxLayout()
        vbox1.addWidget(self.button)
        vbox1.addWidget(self.button1)
        vbox1.addWidget(self.button2)
        vbox1.addWidget(self.button3)

        # Uklad pionowy dla etykiet
        vbox2 = QVBoxLayout()
        vbox2.addWidget(self.label)
        vbox2.addWidget(self.label1)

        # Uklad poziomy dla etykiety i przyciskow
        hbox = QHBoxLayout()
        hbox.addLayout(vbox1)
        hbox.addLayout(vbox2)

        # Uklad pionowy
        vbox = QVBoxLayout()
        vbox.addLayout(hbox)

        # Ustawienie glownego ukladu dla okna
        self.setLayout(vbox)
    """Metoda do wybrania zdjecia i jednoczesnie do wskazania folderu z pozostalymi zdjeciami 
    potrzebnymi do stworzenia modeli"""
    def select_image(self):
        global path
        #Sciezka do zdjecia
        path = eg.fileopenbox()
        #Jezeli pusta to zostanie zastapiona .x
        if path is None:
            path = ".x"
        #Jezeli nie posiada odpowiedniego formatu bedzie otwierac okno do wyboru zdjecia
        while not path.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = eg.fileopenbox()
        else:
            self.button1.setEnabled(True)

        return path

    """Metoda do tworzenia modeli kNN i VGG16"""
    def create_models(self):
        global KNN, knnmodel
        #Stworzenie modelu kNN dla zbioru zdjec
        KNN = k_NN(path_to_dictionary=os.path.dirname(os.path.dirname([path][0])))
        knnmodel = KNN.kNN_model()

        self.button2.setEnabled(True)
        self.button3.setEnabled(True)
        return KNN, knnmodel

    """Metoda do predykcji dla klasyfikatora kNN"""
    def kNN_predict(self):
        #Uzycie metody do predykcji z klasy k_NN dla wybranego zdjecia
        label, predicted_label = KNN.kNN_prediction(model=knnmodel, path_to_image=[path])
        #Wyswietlenie wynikow w interfejsie aplikacji
        self.label.setText("Labeled emotion:" + label[0])
        self.label1.setText("Predicted emotion:" + str(predicted_label))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
