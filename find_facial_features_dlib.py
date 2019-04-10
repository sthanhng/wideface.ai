from __future__ import division
import cv2
import time
import datetime as dt

from imutils.video import VideoStream

from misc import *

from face_api import FaceAPI
from utils.colors import *
import dlib


if __name__ == "__main__":

    # Load config file
    config_path = "configs/config.json"
    config = load_cfg_file(config_path)

    # Config the source to get the stream from the camer/webcam
    source_network = config["source"]["network_src"]
    source = config["source"]["local_src"]

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(config["dlib"][
                                         "shape_predictor_68"])

    cap = cv2.VideoCapture(source)

    frame_count = 0
    tt = 0

    # Create an instance of FaceAPI
    flex_face = FaceAPI(config)
    # vs = VideoStream(src=0).start()
    # vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
    time.sleep(2.0)

    while True:
        _, frame = cap.read()
        frame_count += 1

        start_time = time.time()
        t = time.time()
        frame_dlib = frame.copy()
        frame_dlib = resize(frame_dlib, width=500)
        gray = cv2.cvtColor(frame_dlib, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        for rect in rects:
            # Compute the bounding box of the face and draw it on the frame
            (bX, bY, bW, bH) = rect_to_bb(rect)
            cv2.rectangle(frame_dlib, (bX, bY), (bX + bW, bY + bH),
                          YELLOW1, 2)

            # For every face rectangle, run landmarkDetector
            landmarks = predictor(gray, rect)
            points = shape_to_np(landmarks)

            # Loop over the (x, y) coordinates for the facial landmarks
            # and draw them on the image
            for (i, (x, y)) in enumerate(points):
                cv2.circle(frame_dlib, (x, y), 1, (0, 0, 255), 2)

        tt += time.time() - t
        fps = frame_count / tt

        label = "FLEX FACE | {:.1f} FPS".format(fps)
        cv2.putText(frame_dlib, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, CYAN2, 1, cv2.LINE_AA)
        tss = dt.datetime.now()
        cv2.putText(frame_dlib, tss.strftime('%Y-%m-%d %H:%M:%S'), (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN2, 1,
                    cv2.LINE_AA)

        if frame_count == 1:
            tt = 0

        cv2.imshow("Flex face", frame_dlib)

        k = cv2.waitKey(1)
        if k == 27:
            break
    cv2.destroyAllWindows()

