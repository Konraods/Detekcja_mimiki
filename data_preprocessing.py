#Plik do ustandaryzowania wymiarow zdjec
import cv2

class Preprocessor:
    """Konstruktor z szerokoscia, wysokoscia i interpolacja"""
    def __init__(self, width, height, interpolation=cv2.INTER_AREA):
        #Przechowuje docelowe wymiary do zmienienia rozmiaru obrazow
        self.width = width
        self.height = height
        self.interpolation = interpolation

    """Metoda do zmiany wymiarow zdjec"""
    def preprocessing(self, image):
        return cv2.resize(src=image, dsize=(self.width, self.height), interpolation=self.interpolation)
