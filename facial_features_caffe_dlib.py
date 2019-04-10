import time
import datetime as dt
import dlib

from misc import *
from face_api import FaceAPI
from utils.colors import *

if __name__ == "__main__":

    # Load config file
    config_path = "configs/config.json"
    config = load_cfg_file(config_path)

    # ---------------------------------------------------------------
    #
    # OpenCV DNN supports 2 networks.
    # 1. FP16 version of the original caffe implementation (5.4 MB)
    # 2. 8 bit Quantized version using Tensorflow (2.7 MB)
    #
    # ---------------------------------------------------------------
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
    cap = cv2.VideoCapture(source)

    frame_count = 0
    tt_dlib = 0
    tt_cf = 0

    # Create an instance of FaceAPI
    flex_face = FaceAPI(config, net, landmark_net)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(config["dlib"]["shape_predictor_68"])
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # ---------------------------------------
        #
        # Facial landmarks using dlib
        #
        # ---------------------------------------
        start_time = time.time()
        dlib_outs, rects = flex_face.dlib_detect_face(detector, frame, 1,
                                                      True, YELLOW1)
        dlib_outs = flex_face.dlib_find_landmarks(predictor, dlib_outs, rects,
                                                  True, ALICEBLUE)
        tt_dlib += time.time() - start_time
        fps = frame_count / tt_dlib

        label = "Dlib {:.1f} FPS".format(fps)
        cv2.putText(dlib_outs, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, CYAN2, 1, cv2.LINE_AA)
        tss = dt.datetime.now()
        cv2.putText(dlib_outs, tss.strftime('%Y-%m-%d %H:%M:%S'), (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN2, 1,
                    cv2.LINE_AA)

        # ------------------------------------------
        #
        # Facial landmarks using Landmarks Caffe
        #
        # ------------------------------------------
        start_time = time.time()
        caffe_outs, bboxes, list_caffe_rect = flex_face.detect_face(net,
                                                                      frame)

        LM_caffe_param = 60
        flex_face.caffe_find_landmarks_multi_face(landmark_net, caffe_outs,
                                                  list_caffe_rect,
                                                  LM_caffe_param, True,
                                                  ALICEBLUE)
        tt_cf += time.time() - start_time
        fps = frame_count / tt_cf

        label = "Caffe {:.1f} FPS".format(fps)
        cv2.putText(caffe_outs, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, CYAN2, 1, cv2.LINE_AA)
        tss = dt.datetime.now()
        cv2.putText(caffe_outs, tss.strftime('%Y-%m-%d %H:%M:%S'), (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN2, 1,
                    cv2.LINE_AA)

        combined = np.hstack([caffe_outs, dlib_outs])
        cv2.imshow("Flex face", combined)

        if frame_count == 1:
            tt_dlib = 0
            tt_cf = 0

        k = cv2.waitKey(1)
        if k == 27:
            break
    cv2.destroyAllWindows()
    cap.release()
