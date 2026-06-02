import mediapipe as mp

print("mediapipe version:", mp.__version__)
print("mediapipe location:", mp.__file__)
print("has solutions:", hasattr(mp, "solutions"))
print(dir(mp))