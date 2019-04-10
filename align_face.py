import dlib
import cv2
import time

from misc import resize, rect_to_bb, load_cfg_file
from align.align_face import FaceAligner
from face_api import FaceAPI
from utils.colors import CYAN2, BANANA

# Load config file
config_path = "configs/config.json"
config = load_cfg_file(config_path)

DNN = config["model"]["dnn"]
try:
    if DNN == "CAFFE":
        modelFile = config["model"]["caffe"]["model_file"]
        configFile = config["model"]["caffe"]["config_file"]
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    elif DNN == "CAFFE_FP16":
        modelFile = config["model"]["caffe_fp16"]["model_file"]
        configFile = config["model"]["caffe_fp16"]["config_file"]
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    else:
        modelFile = config["model"]["tensorflow"]["model_file"]
        configFile = config["model"]["tensorflow"]["config_file"]
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
except RuntimeError:
    print("Missing the DNN model!")
    net = None

# Landmarks caffe net
lm_model_file = config["lm_caffe"]["model_file"]
lm_config_file = config["lm_caffe"]["config_file"]
landmark_net = cv2.dnn.readNetFromCaffe(lm_config_file, lm_model_file)

# Config the source to get the stream from the camer/webcam
source = config["source"]["local_src"]
cap = cv2.VideoCapture(0)

frame_count = 0
tt_cf = 0

# Create an instance of FaceAPI
flex_face = FaceAPI(config, net, landmark_net)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(config["dlib"]["shape_predictor_68"])

fa = FaceAligner("caffe", landmark_net, desired_face_width=256)

while True:
    ret, frame = cap.read()
    if not ret:
        print("!!!")
        break
    frame_count += 1

    start_time = time.time()
    caffe_outs, _, list_caffe_rect = flex_face.detect_face(net, frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # image = resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    # loop over the face detections
    for rect in list_caffe_rect:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        # (x, y, w, h) = rect_to_bb(rect)
        # faceOrig = resize(frame[y:y + h, x:x + w], width=256)

        faceOrig = frame[rect[1]:rect[3], rect[0]:rect[2]]
        faceAligned = fa.align(frame, rect, True, BANANA)

    # import uuid

    # f = str(uuid.uuid4())
    # cv2.imwrite("outputs/" + f + ".png", faceAligned)

        cv2.imshow("Original", faceOrig)
        cv2.imshow("Aligned", faceAligned)
    # display the original image
    cv2.imshow("Input", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
cap.release()
