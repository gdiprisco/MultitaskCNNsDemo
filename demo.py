import keras, re, os, sys, time, argparse
from cv2 import cv2 as cv
import numpy as np
from scipy import stats
from keras_vggface import utils
from script.face_detector import FaceDetector
# from deepface.commons import functions 
from script.face_aligner import FaceAligner
from script.multitask_utils import write_str, get_versioned_metrics, custom_objects,\
    get_gender, get_age, get_emotion, get_ethnicity, top_left, bottom_right, bottom_left


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

class MostFrequent:
    
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
        print(self.gender, self.age, self.ethnicity, self.emotion)
        avggender = stats.mode(self.gender.data())[0][0]
        avgage = np.mean(self.age.data())
        avgethnicity = stats.mode(self.ethnicity.data())[0][0]
        avgemotion = stats.mode(self.emotion.data())[0][0]
        print(avggender, avgage, avgethnicity, avgemotion)
        return avggender, avgage, avgethnicity, avgemotion

    def restore(self):
        self.gender.restore()
        self.age.restore()
        self.ethnicity.restore()
        self.emotion.restore()

def main(source, destination=None, gpu=None, movingaverage=False, framedelay=None):
    print("Reading",  source if source != 0 else "from internal cam...")
    cam = cv.VideoCapture(source)
    if destination is not None:
        width  = int(cam.get(3))
        height = int(cam.get(4))
        out = cv.VideoWriter(destination, cv.VideoWriter_fourcc(*'MP4V'), 20.0, (width,height))
    MTN = MultiTaskNetwork()
    facedet = FaceDetector(conf_thresh=0.65)
    facealign = FaceAligner()
    color = (9, 87, 39)
    mostfrequent, use_mostfrequent = MostFrequent(5, 5, 5, 5), True
    frame = 0
    while True:
        try:
            _, annImage = cam.read()
            faces = facedet.detect(annImage)
            if len(faces) != 1 or realtime:
                use_mostfrequent = False
                mostfrequent.restore()
            else:
                use_mostfrequent = True
            if frame >= framedelay or framedelay is None:
                for f in faces:
                    aligned_face = (facealign.align(f['img'], (0,0,f['img'].shape[0],f['img'].shape[1])))
                    G, A, E, R = MTN.get_prediction(aligned_face)
                    if use_mostfrequent:
                        G, A, E, R = mostfrequent.average(G, A, E, R)
                    cv.rectangle(annImage, top_left(f), bottom_right(f), color, 2)
                    write_str(annImage, "%s\n%d years\n%s\n%s" % (G, A, E, R), bottom_left(f), color)
            if destination is None:
                cv.imshow('Multitask CNNs for efficient face analysis in the wild',annImage)
            else:
                out.write(annImage)
            if cv.waitKey(1) == ord('q'):
                cv.destroyAllWindows()
                exit()
        except Exception as e:
            print("Exception:", e)
        finally:
            frame += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multitask CNNs - Di Prisco Giovanni')
    parser.add_argument("--gpu", type=int, dest="gpu", required=False, help="GPU selected")
    parser.add_argument("--input", type=str, dest="input", required=False, help="Source. If not specified internal cam is used.")
    parser.add_argument("--output", type=int, dest="output", required=False, help="Destination. If not specified opencv show is used.")
    parser.add_argument("--movingaverage", type=int, dest="movingaverage", default=False, help="Moving Average for samples.")
    parser.add_argument("--framedelay", type=int, dest="framedelay", required=False, help="Frame delay for multitask ")
    args = parser.parse_args()
    if gpu is not none:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    main(args.input, args.output, args.gpu, args.movingaverage, args, framedelay)