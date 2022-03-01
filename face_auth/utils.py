
import base64
import io

import cv2
import face_recognition
import numpy as np
from PIL import Image


def str_to_pil_image(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))


def pil_to_cv2_image(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)


def cv2_img_to_encodings(face):
    face_bounding_boxes = face_recognition.face_locations(face)
    if len(face_bounding_boxes) == 1:
        return face_recognition.face_encodings(face)[0]
    return None


