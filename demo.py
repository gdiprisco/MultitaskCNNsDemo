import argparse
import os
import re
import sys
import time

import keras
import numpy as np
from cv2 import cv2 as cv
from keras_vggface import utils
from scipy import stats

# from deepface.commons import functions 
from script.face_aligner import FaceAligner
from script.face_detector import FaceDetector
from script.videotracker import CentroidTracker
from script.multitask_utils import (bottom_left, bottom_right, custom_objects,
                                    get_age, get_emotion, get_ethnicity,
                                    get_gender, get_versioned_metrics, bottom_center,
                                    top_left, write_str, get_gender_cat, coords)

MODEL_PATH = "models/_netresnet50_versionverC_pretrainingimagenet_datasetVGGFace2-RAF_preprocessingvggface2_augmentationdefault_batch64_lr0.001_0.5_120_sel_gpu2_training-epochs400_20210102_154443/checkpoint.600.hdf5"

class MultiTaskNetwork:

    def __init__(self, model_path=MODEL_PATH):
        print ("MultiTask -> init")
        self.model, self.INPUT_SHAPE = self._load_keras_model(MODEL_PATH)
        print ("MultiTask -> init ok")

    def _load_keras_model(self,filepath):
        version = re.search("versionver[ABC]", os.path.split(os.path.split(filepath)[0])[1])
        if not version:
            raise Exception("Unable to infer model version from path splitting")
        version = version[0].replace("version", "")
        loss, loss_weights, accuracy_metrics, _ = get_versioned_metrics(version)
        model = keras.models.load_model(filepath, custom_objects=custom_objects, compile=False)
        model.compile(loss=loss, loss_weights=loss_weights, optimizer='sgd', metrics=accuracy_metrics)
        INPUT_SHAPE = (112, 112, 3)
        return model, INPUT_SHAPE

    def _preprocess(self,face):
        # RESHAPE
        face = cv.resize(face, (112, 112), interpolation=cv.INTER_CUBIC)
        # BGR 2 RGB
        face_rgb = np.expand_dims(cv.cvtColor(face, cv.COLOR_BGR2RGB), 0)
        # img - mean
        face_rgb = utils.preprocess_input(face_rgb.astype(np.float32))
        return face_rgb


    def get_prediction(self,image):
        res = self.model.predict(self._preprocess(image))
        gender = get_gender(res[-4][0])
        age = get_age(res[-3][0][0])
        ethnicity = get_ethnicity(res[-2][0])
        emotion = get_emotion(res[-1][0])
        return [gender, age, ethnicity, emotion]

class Heap:
    def __init__(self, size:int):
        self.array = list()
        self.size = size
    
    def append(self, element):
        if len(self.array) >= self.size:
            del self.array[0]
        self.array.append(element)

    def restore(self):
        self.array = list()

    def data(self):
        return self.array

    def __iter__(self):
        return self.array.__iter__()

class MovingAverage:
    
    __slots__ = "gender", "age", "ethnicity", "emotion",\
        "gender_samples", "age_samples", "ethnicity_samples", "emotion_samples"

    def __init__(self, gender_samples:int, age_samples:int, ethnicity_samples:int, emotion_samples:int):
        self.gender = Heap(gender_samples)
        self.age = Heap(age_samples)
        self.ethnicity = Heap(ethnicity_samples)
        self.emotion = Heap(emotion_samples)
    
    def average(self, gender, age, ethnicity, emotion):
        self.gender.append(gender)
        self.age.append(age)
        self.ethnicity.append(ethnicity)
        self.emotion.append(emotion)
        print(self.gender.array)
        print(self.age.array)
        print(self.ethnicity.array)
        print(self.emotion.array)
        print()
        avggender = stats.mode(self.gender.data())[0][0]
        avgage = np.mean(self.age.data())
        avgethnicity = stats.mode(self.ethnicity.data())[0][0]
        avgemotion = stats.mode(self.emotion.data())[0][0]
        return avggender, avgage, avgethnicity, avgemotion

    def restore(self):
        self.gender.restore()
        self.age.restore()
        self.ethnicity.restore()
        self.emotion.restore()

class MultiMovingAverage:
    __slots__ = "faces", "gender_samples", "age_samples", "ethnicity_samples", "emotion_samples"

    def __init__(self, gender_samples:int, age_samples:int, ethnicity_samples:int, emotion_samples:int):
        self.faces = {}
        self.gender_samples = gender_samples
        self.age_samples = age_samples
        self.ethnicity_samples = ethnicity_samples
        self.emotion_samples = emotion_samples

    def _register(self, identifier):
        self.faces[identifier] = MovingAverage(self.gender_samples,\
                self.age_samples, self.ethnicity_samples, self.emotion_samples)

    # TODO deregister update: counter on average for saving memory

    def _deregister(self, identifier):
        del self.faces[identifier]

    def average(self, identifier, gender, age, ethnicity, emotion):
        print("FaceID", identifier)
        if identifier not in self.faces:
            self._register(identifier)
        return self.faces[identifier].average(gender, age, ethnicity, emotion)


STRCOLORS = [(122,64,236),(243,150,33)]
def select_strcolor(gender):
    return STRCOLORS[get_gender_cat(gender)]


debug_movingaverage = True
CONFIDENCE_DETECTOR = 0.5#0.65
GMA, AMA, EtMA, EmMA = 10, 200, 20, 100 #10, 120, 20, 40 
CENTROID_TOLERANCE = 3
bounding_color = (3, 255, 118)
label_color = (255, 255, 255) #(3, 255, 118) #(9, 87, 39)
text_color = (0, 0, 0)

def main(source, destination=None, movingaverage=False, alignment=False, framedelay=None):
    print("-------------- PARAMETERS --------------")
    print("Reading",  source if source != 0 else "from internal cam...")
    print("Output:", "imshow" if destination is None else destination)
    print("Moving Average:", movingaverage)
    print("Face Alignment:", alignment)
    print("Frame Delay:", framedelay)
    print("----------------------------------------")
    cam = cv.VideoCapture(source)
    W, H = int(cam.get(3)), int(cam.get(4))
    fontsize = int(H//20.5) #35
    if destination is not None:
        out = cv.VideoWriter(destination, cv.VideoWriter_fourcc(*'MP4V'), 20.0, (W,H))
    multitask = MultiTaskNetwork()
    facedet = FaceDetector(conf_thresh=CONFIDENCE_DETECTOR)
    facealign = FaceAligner()

    tracker = CentroidTracker(maxDisappeared=CENTROID_TOLERANCE)

    if movingaverage or debug_movingaverage:
        averager = MultiMovingAverage(GMA, AMA, EtMA, EmMA)
    frame = 0
    while True:
        try:
            _, annImage = cam.read()
            faces = facedet.detect(annImage)
            rects = []

            if framedelay is None or frame >= framedelay:
                if movingaverage:
                    for f in faces:
                        rects.append(coords(f))
                    objects, items = tracker.update(rects, faces)
                    for ((faceID, centroid), (_, f)) in zip(objects.items(), items.items()):
                        if alignment:
                            face_coords = (0,0,f['img'].shape[0],f['img'].shape[1])
                            face = (facealign.align(f['img'], face_coords)) 
                        else:
                            face = f['img']
                        G, A, E, R = multitask.get_prediction(face)
                        if movingaverage:
                            G, A, E, R = averager.average(faceID, G, A, E, R)
                        cv.rectangle(annImage, top_left(f), bottom_right(f), bounding_color, 2)
                        label_apply = bottom_center(f) #bottom_left(f) #left alignment
                        annImage = write_str(annImage, "%s, %d\n%s\n%s" % (G, A, E, R), label_apply, label_color, (select_strcolor(G), text_color), fontsize)

                        # text = "ID {}".format(faceID)
                        # cv.putText(annImage, text, (centroid[0] - 10, centroid[1] - 10),
                        #     cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        # cv.circle(annImage, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                else:
                    for f in faces:
                        if alignment:
                            face_coords = (0,0,f['img'].shape[0],f['img'].shape[1])
                            face = (facealign.align(f['img'], face_coords)) 
                        else:
                            face = f['img']
                        G, A, E, R = multitask.get_prediction(face)
                        if debug_movingaverage:
                            G, A, E, R = averager.average(0, G, A, E, R)
                        cv.rectangle(annImage, top_left(f), bottom_right(f), bounding_color, 2)
                        label_apply = bottom_center(f) #bottom_left(f)
                        annImage = write_str(annImage, "%s, %d\n%s\n%s" % (G, A, E, R), label_apply, label_color, (select_strcolor(G), text_color), fontsize)

            if destination is None:
                cv.imshow('Multitask CNNs for efficient face analysis in the wild',annImage)
            else:
                out.write(annImage)
            if cv.waitKey(1) == ord('q'):
                cv.destroyAllWindows()
                exit()
        # except Exception as e:
        #     print("Exception:", e)
        finally:
            frame += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multitask CNNs - Di Prisco Giovanni')
    parser.add_argument("--gpu", type=int, dest="gpu", required=False, help="GPU selected")
    parser.add_argument("--input", type=str, dest="input", required=False, help="Source. If not specified internal cam is used.")
    parser.add_argument("--output", type=str, dest="output", required=False, help="Destination. If not specified opencv show is used.")
    parser.add_argument("--movingaverage", action="store_true", dest="movingaverage", help="Moving Average for samples.")
    parser.add_argument("--alignment", action="store_true", dest="alignment", help="Perform face alignment.")
    parser.add_argument("--framedelay", type=int, dest="framedelay", required=False, help="Frame delay for multitask ")
    args = parser.parse_args()
    print("++++++++++++++++++++++++++++++++++++++++")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        print("Using GPU", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("USING CPU ENGINE - NO TENSORFLOW GPU AVAILABLE OR NO GPU PARAM SPECIFIED")
    print("++++++++++++++++++++++++++++++++++++++++")
    main(args.input if args.input is not None else 0, args.output, args.movingaverage, args.alignment, args.framedelay)

    # demo.py --input video-in/JackieChan-ChrisTucker.mp4 --output video-out/JackieChan-ChrisTucker.mp4 --movingaverage
    # demo.py --movingaverage
    # demo.py --movingaverage --framedelay 3