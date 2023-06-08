#Plik do wykorzystania modulu mediapipe do nakladania siatki na twarze
import cv2
import mediapipe as mp
from datetime import datetime
class Preprocessor:
    """Konstruktor z szerokoscia, wysokoscia i interpolacja"""
    def __init__(self, width, height, interpolation=cv2.INTER_AREA):
        #Przechowuje docelowe wymiary do zmienienia rozmiaru obrazow
        self.width = width
        self.height = height
        self.interpolation = interpolation
    """
    def face_image(self, image):
        global face

        # Wyciecie twarzy z calego zdjecia
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'Haarcascade_frontalface_default.xml')
        face = haar_cascade.detectMultiScale(gray_img)

        return face
    """
    """Metoda do zmiany wymiarow zdjec"""
    def preprocessing(self, image):

        # Do nalozenia mesh'a na twarze za pomoca Mediapipe
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 0, 255))

        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.40) as face_mesh:
            """
            self.face_image(image)

            for (x, y, w, h) in face:
                faces = image[y:y + h+10, x:x + w+10]
                image = faces
            """
            # Zmiana z formatu BGR do RGB
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                pass
            annotated_image = image.copy()
            original_image_resized = cv2.resize(src=image, dsize=(self.width, self.height), interpolation=self.interpolation)
            annotated_image = cv2.resize(src=annotated_image, dsize=(self.width, self.height), interpolation=self.interpolation)

            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=0),
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )
        return annotated_image