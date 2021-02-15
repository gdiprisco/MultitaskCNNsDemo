import os, dlib
from cv2 import cv2 as cv
from imutils.face_utils import FaceAligner as FA

PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "models")
shape_predictor = os.path.join(PATH, "shape_predictor_68_face_landmarks.dat")

class FaceAligner:

    __slots__ = ["face_aligner"]

    def __init__(self):
        self.face_aligner = FA(predictor=dlib.shape_predictor(shape_predictor), desiredFaceWidth=224)

    def align(self, image, roi):
        rgb = image.copy()
        grey = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)
        left, top, right, bottom = roi
        r = dlib.rectangle(left, top, right, bottom)
        return self.face_aligner.align(rgb, grey, r)