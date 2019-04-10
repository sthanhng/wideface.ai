import argparse
import dlib
import cv2

from misc import resize, rect_to_bb
from align.face_aligner_dlib import FaceAligner

parse = argparse.ArgumentParser()
parse.add_argument("--shape-predictor", type=str,
                   default="dlib/shape_predictor_68_face_landmarks.dat",
                   help="path to facial landmark predictor")
args = parse.parse_args()

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.shape_predictor)

fa = FaceAligner(predictor, desired_face_width=256)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("!!!")
        break

    # image = resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    # loop over the face detections
    for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)
        faceOrig = resize(frame[y:y + h, x:x + w], width=256)
        faceAligned = fa.align(frame, gray, rect)

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
