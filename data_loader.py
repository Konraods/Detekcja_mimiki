#Plik do wczytania zdjec
import os
import cv2
import random
import shutil
import numpy as np

class DataLoader:
    def __init__(self, preprocessor=None):
        #Przechowuje preprocesor
        self.preprocessor = preprocessor

        #Bez podania preprocesora inicjuje pusta liste
        if self.preprocessor is None:
            self.preprocessor = []

    def data_load_kNN(self, input_path):
        #Inicjuje listy na etykiety i zdjecia
        labels = []
        images = []

        for idx, input_path in enumerate(input_path):
            image = cv2.imread(input_path)
            # Do kazdego zdjecia aplikowany jest preprocesor
            if self.preprocessor is not None:
                for process in self.preprocessor:
                    image = process.preprocessing(image)

            #Dodanie do list etykiet i zdjec w postaci wektorow
            labels.append(input_path.split(os.path.sep)[-2])
            images.append(image)

        #Zwrot krotki w postaci wektora i etykiety
        return (np.array(images), np.array(labels))

    def data_load_VGG(self, input_path, emotions):
        for idx, input_path in enumerate(input_path):
            image = cv2.imread(input_path)
            if self.preprocessor is not None:
                for process in self.preprocessor:
                    image = process.preprocessing(image)
                    cv2.imwrite(f"{input_path}", image)

        # Tworzymy foldery "train" i "test"
        try:
            os.makedirs("train")
            os.makedirs("test")
        except OSError as err:
            print(f"Error {err}")

        # Tworzymy podfoldery w folderach "train" i "train"
        for folder in emotions:
            try:
                os.makedirs("train/" + folder)
                os.makedirs("test/" + folder)
            except OSError as err:
                print(f"Error {err}")

        # Dzielimy pliki w podfolderach na dwie grupy: 80% dla "train" i 20% dla "train"
        for folder in emotions:
            files = os.listdir("input/" + folder)
            split = int(0.8 * len(files))
            train_files = random.sample(files, split)
            test_files = [f for f in files if f not in train_files]

            # Przenosimy pliki do odpowiednich podfolderów
            for file in train_files:
                src = "input/" + folder + "/" + file
                dst = "train/" + folder + "/" + file
                shutil.copyfile(src, dst)

            for file in test_files:
                src = "input/" + folder + "/" + file
                dst = "test/" + folder + "/" + file
                shutil.copyfile(src, dst)



