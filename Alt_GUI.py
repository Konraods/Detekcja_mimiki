#Aplikacja okienkowa do obslugi klasyfikatora kNN i sieci neuronowej VGG16
import sys
import os
import easygui as eg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout
from kNN import k_NN
from VGG16 import VGG_16

class MainWindow(QWidget):
    def __init__(self):
        self.V1 = None
        self.V2 = None
        self.path = ""
        super().__init__()

        # Ustawienia okna
        self.setGeometry(150, 150, 600, 300)
        self.setWindowTitle('Aplikacja do detekcji mimiki twarzy')

        # Przycisk - zdjecia
        self.button = QPushButton('Select image', self)
        self.button.clicked.connect(self.select_image)

        # Przycisk - create knn model
        self.button1 = QPushButton('Create k-NN model', self)
        self.button1.setEnabled(False)
        self.button1.clicked.connect(self.create_knnmodel)

        # Przycisk - create vgg16 model
        self.button2 = QPushButton('Create VGG16 model', self)
        self.button2.setEnabled(False)
        self.button2.clicked.connect(self.create_vgg16model)

        # Przycisk - k-nn
        self.button3 = QPushButton('Prediction', self)
        self.button3.setEnabled(False)
        self.button3.clicked.connect(self.kNN_predict)

        # Przycisk - vgg16
        self.button4 = QPushButton('Prediction', self)
        self.button4.setEnabled(False)
        self.button4.clicked.connect(self.VGG16_predict)

        # Przycisk - load vgg16 model
        self.button5 = QPushButton('Load VGG16 model', self)
        self.button5.clicked.connect(self.load_vgg16model)

        # Etykieta - kNN
        self.label = QLabel("<b>{}</b>".format("k-NN"), self)
        self.label.setAlignment(Qt.AlignCenter)

        # Etykieta - klasa
        self.label1 = QLabel('Labeled emotion: ', self)
        self.label1.setAlignment(Qt.AlignCenter)

        # Etykieta - przewidywania
        self.label2 = QLabel('Predicted emotion:', self)
        self.label2.setAlignment(Qt.AlignCenter)

        # Etykieta - VGG16
        self.label3 = QLabel("<b>{}</b>".format("VGG16"), self)
        self.label3.setAlignment(Qt.AlignCenter)

        # Etykieta - klasa
        self.label4 = QLabel('Labeled emotion: ', self)
        self.label4.setAlignment(Qt.AlignCenter)

        # Etykieta - przewidywania
        self.label5 = QLabel('Predicted emotion:', self)
        self.label5.setAlignment(Qt.AlignCenter)

        # Uklad poziomy dla label knn i vgg16
        hbox = QHBoxLayout()
        hbox.addWidget(self.label)
        hbox.addWidget(self.label3)

        # Uklad poziomy dla przyciskow create vgg16 i load vgg16
        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.button2)
        hbox1.addWidget(self.button5)

        # Uklad poziomy dla przyciskow create knn i uklady przyciskow vgg16
        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.button1)
        hbox2.addLayout(hbox1)

        # Uklad poziomy dla przyciskow create vgg16 i load vgg16
        hbox3 = QHBoxLayout()
        hbox3.addWidget(self.button3)
        hbox3.addWidget(self.button4)

        # Uklad poziomy dla labels labeled
        hbox4 = QHBoxLayout()
        hbox4.addWidget(self.label1)
        hbox4.addWidget(self.label4)

        # Uklad poziomy dla labels predicted
        hbox5 = QHBoxLayout()
        hbox5.addWidget(self.label2)
        hbox5.addWidget(self.label5)

        #Uklad pionowy dla wszystkich
        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(self.button)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        vbox.addLayout(hbox4)
        vbox.addLayout(hbox5)

        # Glowny uklad
        self.setLayout(vbox)

    """Metoda do wybrania zdjecia i jednoczesnie do wskazania folderu z pozostalymi zdjeciami 
    potrzebnymi do stworzenia modeli"""
    def select_image(self):
        #Sciezka do zdjecia
        self.path = eg.fileopenbox()
        #Jezeli pusta to zostanie zastapiona .x
        if self.path is None:
            self.path = ".x"
        #Jezeli nie posiada odpowiedniego formatu bedzie otwierac okno do wyboru zdjecia
        while not self.path.lower().endswith(('.png', '.jpg', '.jpeg')):
            self.path = eg.fileopenbox()
        else:
            self.button1.setEnabled(True)
            self.button2.setEnabled(True)
        return self.path

    """Metoda do tworzenia modeli kNN i VGG16"""
    def create_knnmodel(self):
        global KNN, knnmodel
        #Stworzenie modelu kNN dla zbioru zdjec
        KNN = k_NN(path_to_dictionary=os.path.dirname(os.path.dirname([self.path][0])))
        knnmodel = KNN.kNN_model()

        self.button3.setEnabled(True)

        return KNN, knnmodel

    """Metoda do wczytania modelu sieci neuronowej VGG16"""
    def create_vgg16model(self):
        global VGG16, vgg16model

        VGG16 = VGG_16(path_to_dictionary=os.path.dirname(os.path.dirname([self.path][0])))
        VGG16.delete_dictionaries()
        vgg16model = VGG16.vgg16_model()

        self.button4.setEnabled(True)
        self.V1 = 1
        return VGG16, vgg16model

    def load_vgg16model(self):
        global VGG16_loaded, model_loaded

        model_loaded = eg.fileopenbox()
        if model_loaded is None:
            model_loaded = ".x"
        #Jezeli nie posiada odpowiedniego formatu bedzie otwierac okno do wyboru zdjecia
        while not model_loaded.lower().endswith('.h5'):
            model_loaded = eg.fileopenbox()
        else:
            self.button4.setEnabled(True)

        VGG16_loaded = VGG_16(path_to_dictionary=None)
        self.V2 = 1
        return VGG16_loaded, model_loaded

    """Metoda do predykcji dla klasyfikatora kNN"""
    def kNN_predict(self):
        #Uzycie metody do predykcji z klasy k_NN dla wybranego zdjecia
        label, predicted_label = KNN.kNN_prediction(model=knnmodel, path_to_image=[self.path])
        #Wyswietlenie wynikow w interfejsie aplikacji
        self.label1.setText("Labeled emotion: <b>{}</b>".format(label[0]))
        self.label2.setText("Predicted emotion: <b>{}</b>".format(predicted_label))

    """Metoda do predykcji dla sieci neuronowej VGG16"""
    def VGG16_predict(self):
        if self.V1 is not None:
            predicted_label = VGG16.prediction(nnmodel=[vgg16model], path_to_image=[self.path])

            self.label4.setText("Labeled emotion: <b>{}</b>".format(self.path.split("\\")[-2]))
            self.label5.setText("Predicted emotion:  <b>{}</b>".format(predicted_label))

        if self.V2 is not None:
            if self.path == "":
                self.select_image()
            predicted_label = VGG16_loaded.prediction(nnmodel=[model_loaded], path_to_image=[self.path])

            self.label4.setText("Labeled emotion: <b>{}</b>".format(self.path.split("\\")[-2]))
            self.label5.setText("Predicted emotion:  <b>{}</b>".format(predicted_label))




if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
