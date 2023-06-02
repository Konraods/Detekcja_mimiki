import sys

#import VGG16
import kNN
import easygui as eg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout
from data_loader import DataLoader
from data_preprocessing import Preprocessor


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Ustawienia okna
        self.setGeometry(150, 150, 450, 150)
        self.setWindowTitle('Aplikacja do detekcji mimiki twarzy')

        # Przycisk - zdjecia
        self.button = QPushButton('Zdjęcie', self)
        self.button.clicked.connect(self.select_image)

        # Przycisk - wideo
        self.button1 = QPushButton('Wideo - nie robiłem jeszcze', self)
        #self.button1.clicked.connect(self.on_button_click)

        # Przycisk - k-nn
        self.button2 = QPushButton('Klasyfikator k-NN', self)
        self.button2.setEnabled(False)
        self.button2.clicked.connect(self.k_NN)

        # Przycisk - vgg16
        self.button3 = QPushButton('Sieć neuronowa VGG16', self)
        self.button3.setEnabled(False)
        #self.button3.clicked.connect(self.on_button_click)

        # Etykieta - klasa
        self.label = QLabel('Prawdziwe emocje: ', self)
        self.label.setAlignment(Qt.AlignCenter)

        # Etykieta - przewidywania
        self.label1 = QLabel('Przewidywane emocje:', self)
        self.label1.setAlignment(Qt.AlignCenter)

        # Układ pionowy dla przycisków
        vbox1 = QVBoxLayout()
        vbox1.addWidget(self.button)
        vbox1.addWidget(self.button1)
        vbox1.addWidget(self.button2)
        vbox1.addWidget(self.button3)

        # Układ pionowy dla etykiet
        vbox2 = QVBoxLayout()
        vbox2.addWidget(self.label)
        vbox2.addWidget(self.label1)

        # Układ poziomy dla etykiety i przycisków
        hbox = QHBoxLayout()
        hbox.addLayout(vbox1)
        hbox.addLayout(vbox2)

        # Układ pionowy
        vbox = QVBoxLayout()
        vbox.addLayout(hbox)

        # Ustawienie głównego układu dla okna
        self.setLayout(vbox)

    def select_image(self):
        global path
        path = eg.fileopenbox()
        self.button2.setEnabled(True)
        self.button3.setEnabled(True)
        return path

    def k_NN(self):
        temp = [path]
        preproc = Preprocessor(224, 224)
        loader = DataLoader(preprocessor=[preproc])
        (images, labels) = loader.data_load_kNN(temp)
        image = images.reshape((images.shape[0], 224 * 224 * 3))
        model = kNN.model
        prediction = model.predict(image)[0]
        predicted_label = (list(map(kNN.class_mapping.get, [prediction])))
        self.label.setText("Prawdziwe emocje:" + labels[0])
        self.label1.setText("Przewidywane emocje:" + str(predicted_label[0]))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
