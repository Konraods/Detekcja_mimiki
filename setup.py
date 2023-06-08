from distutils.core import setup
import py2exe

setup(
    name="Aplikacja do detekcji mimiki twarzy",
    version="1.0",
    description="Aplikacja do detekcji mimiki twarzy za pomocą Mediapipe",
    windows=["GUI.py"],
)
