import argparse
import dlib
import cv2

from misc import resize, rect_to_bb
from align.face_aligner_dlib import FaceAligner

parse = argparse.ArgumentParser()
parse.add_argument("--shape-predictor", required=True,
                help="path to facial landmark predictor")
parse.add_argument("--image", required=True,
                help="path to input image")
args = parse.parse_args()

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.shape_predictor)

fa = FaceAligner(predictor, desired_face_width=256)

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args.image)
image = resize(image, width=800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Input", image)
rects = detector(gray, 1)

# loop over the face detections
for rect in rects:
    # extract the ROI of the *original* face, then align the face
    # using facial landmarks
    (x, y, w, h) = rect_to_bb(rect)
    faceOrig = resize(image[y:y + h, x:x + w], width=256)
    faceAligned = fa.align(image, gray, rect)

    import uuid

    f = str(uuid.uuid4())
    cv2.imwrite("outputs/" + f + ".png", faceAligned)

    cv2.imshow("Original", faceOrig)
    cv2.imshow("Aligned", faceAligned)
    cv2.waitKey(0)
