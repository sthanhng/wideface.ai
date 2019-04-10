import dlib

from misc import *
from face_api import FaceAPI
from align.align_face import FaceAligner


if __name__ == "__main__":

    # Load config file
    config_path = "configs/config.json"
    config = load_cfg_file(config_path)

    DNN = config["model"]["dnn"]
    try:
        if DNN == "CAFFE":
            modelFile = config["model"]["caffe"]["model_file"]
            configFile = config["model"]["caffe"]["config_file"]
            net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        else:
            modelFile = config["model"]["tensorflow"]["model_file"]
            configFile = config["model"]["tensorflow"]["config_file"]
            net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    except RuntimeError:
        print("Missing the DNN model!")
        net = None

    shape_predictor = "dlib/shape_predictor_68_face_landmarks.dat"
    landmarks_predictor = dlib.shape_predictor(shape_predictor)

    # Landmarks caffe net
    lm_model_file = config["lm_caffe"]["model_file"]
    lm_config_file = config["lm_caffe"]["config_file"]
    landmark_net = cv2.dnn.readNetFromCaffe(lm_config_file, lm_model_file)

    # Config the source to get the stream from the camer/webcam
    source_network = config["source"]["network_src"]
    source = config["source"]["local_src"]

    FaceAlign = FaceAligner("caffe", landmark_net)
    FlexFace = FaceAPI(config, net, landmark_net, FaceAlign)

    # Get datasset
    FlexFace.get_dataset_from_webcam(source, save_video=True, landmarks=False)
