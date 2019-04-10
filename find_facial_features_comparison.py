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

    # Load YOLO face model
    model_cfg = config["yolo"]["config_file"]
    model_weights = config["yolo"]["model_file"]
    yolo_net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
    yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Landmarks caffe net
    lm_model_file = config["lm_caffe"]["model_file"]
    lm_config_file = config["lm_caffe"]["config_file"]
    landmark_net = cv2.dnn.readNetFromCaffe(lm_config_file, lm_model_file)

    # Config the source to get the stream from the camer/webcam
    source = config["source"]["local_src"]
    cap = cv2.VideoCapture(source)

    frame_count = 0
    tt = 0
    tt_dlib = 0
    tt_cf = 0
    tt_yolo = 0

    # Create an instance of FaceAPI
    flex_face = FaceAPI(config)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(config["dlib"]["shape_predictor_68"])
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # ------------------------------------------
        #
        # Facial landmarks using OpenCV DNN
        #
        # ------------------------------------------
        start_time = time.time()
        t = time.time()
        ocv_outs, _, list_dlib_rect, _ = flex_face.detect_face(net, frame)
        for rect in list_dlib_rect:
            # For every face rectangle, run landmarkDetector
            landmarks = predictor(frame, rect)
            points = landmarks.parts()
            for p in points:
                cv2.circle(ocv_outs, (p.x, p.y), 1, ALICEBLUE, 2)
        tt += time.time() - t
        fps = frame_count / tt

        label = "Flex Face {:.1f} FPS".format(fps)
        cv2.putText(ocv_outs, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, CYAN2, 1, cv2.LINE_AA)
        tss = dt.datetime.now()
        cv2.putText(ocv_outs, tss.strftime('%Y-%m-%d %H:%M:%S'), (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN2, 1,
                    cv2.LINE_AA)

        # ---------------------------------------
        #
        # Facial landmarks using dlib
        #
        # ---------------------------------------
        start_time = time.time()
        t = time.time()
        dlib_outs, _ = flex_face.dlib_find_landmarks_custom(frame,
                                                                detector,
                                                                predictor)
        tt_dlib += time.time() - t
        fps = frame_count / tt_dlib

        label = "Dlib {:.1f} FPS".format(fps)
        cv2.putText(dlib_outs, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, CYAN2, 1, cv2.LINE_AA)
        tss = dt.datetime.now()
        cv2.putText(dlib_outs, tss.strftime('%Y-%m-%d %H:%M:%S'), (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN2, 1,
                    cv2.LINE_AA)

        # ---------------------------------------
        #
        #  YOLO Face
        #
        # ---------------------------------------
        start_time = time.time()
        t = time.time()
        yolo_outs, _, list_yolo_rect = flex_face.yolo_face_detect(yolo_net,
                                                                  frame)
        tt_yolo += time.time() - t
        fps = frame_count / tt_yolo

        # Loop over the face detections
        for rect in list_yolo_rect:
            # For every face rectangle, run landmarkDetector
            landmarks = predictor(yolo_outs, rect)
            points = landmarks.parts()

            for p in points:
                cv2.circle(yolo_outs, (p.x, p.y), 1, ALICEBLUE, 2)

        label = "YOLO Face {:.1f} FPS".format(fps)
        cv2.putText(yolo_outs, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, CYAN2, 1, cv2.LINE_AA)
        tss = dt.datetime.now()
        cv2.putText(yolo_outs, tss.strftime('%Y-%m-%d %H:%M:%S'), (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN2, 1,
                    cv2.LINE_AA)

        # ------------------------------------------
        #
        # Facial landmarks using Landmarks Caffe
        #
        # ------------------------------------------
        start_time = time.time()
        t = time.time()
        caffe_outs, _, _, list_caffe_rect = flex_face.detect_face(net, frame)

        LM_caffe_param = 60
        list_CLM = []
        for rect in list_caffe_rect:
            left, top, right, bottom = rect
            roi = caffe_outs[top:bottom + 1, left:right + 1]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            res = cv2.resize(gray_roi, (LM_caffe_param,
                                        LM_caffe_param)).astype(np.float32)

            m = np.zeros((LM_caffe_param, LM_caffe_param))
            std_dev = np.zeros((LM_caffe_param, LM_caffe_param))
            mean, std_dev = cv2.meanStdDev(res, m, std_dev)
            normalized_roi = (res - mean[0][0]) / (0.000001 + std_dev[0][0])

            # For every face rectangle, run landmarkDetector
            blob = cv2.dnn.blobFromImage(normalized_roi, 1.0,
                                         (LM_caffe_param, LM_caffe_param), None)
            landmark_net.setInput(blob)
            caffe_landmarks = landmark_net.forward()

            for landmark in caffe_landmarks:
                LM = []
                for i in range(len(landmark) // 2):
                    x = landmark[2 * i] * (right - left) + left
                    y = landmark[2 * i + 1] * (bottom - top) + top
                    LM.append((int(x), int(y)))
                list_CLM.append(LM)

            for lm in list_CLM:
                for idx, p in enumerate(lm):
                    cv2.circle(caffe_outs, p, 1, ALICEBLUE, 2)

        tt_cf += time.time() - t
        fps = frame_count / tt_cf

        label = "Caffe {:.1f} FPS".format(fps)
        cv2.putText(caffe_outs, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, CYAN2, 1, cv2.LINE_AA)
        tss = dt.datetime.now()
        cv2.putText(caffe_outs, tss.strftime('%Y-%m-%d %H:%M:%S'), (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN2, 1,
                    cv2.LINE_AA)

        top = np.hstack([ocv_outs, dlib_outs])
        bottom = np.hstack([caffe_outs, yolo_outs])
        combined = np.vstack([top, bottom])
        cv2.imshow("Find Facial Features Comparison", combined)

        if frame_count == 1:
            tt = 0
            tt_dlib = 0
            tt_cf = 0
            tt_yolo = 0

        # cv2.imshow("Flex face", outs)
        k = cv2.waitKey(1)
        if k == 27:
            break
    cv2.destroyAllWindows()
    cap.release()
