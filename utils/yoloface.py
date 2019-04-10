import datetime
import cv2
import numpy as np

import dlib

from utils.colors import YELLOW1
from misc import custom_draw_bb


# ----------------------------------------------------
# Parameters
# ----------------------------------------------------
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416

# Default colors
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 215, 255)

# Modes
modes = {
    0: 'detect',
    1: 'track'
}

# Tracking mode
NUM_TRACKING_FRAMES = 5


# ----------------------------------------------------
# Helper functions
# ----------------------------------------------------

# Get the names of the output layers
def get_outputs_names(net):
    # Get the names of all the layers in the network
    layers_names = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected
    # outputs
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def draw_predict(frame, conf, left, top, right, bottom):
    # Draw a bounding box.
    # cv2.rectangle(frame, (left, top), (right, bottom), YELLOW1, 2)
    custom_draw_bb(frame, left, top, right - left, bottom - top, YELLOW1, 2)


def post_process(frame, outs, conf_threshold, nms_threshold):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.
    confidences = []
    boxes = []
    final_boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant
    # overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                               nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        final_boxes.append(box)
        draw_predict(frame, confidences[i], left, top, left + width,
                     top + height)
    return final_boxes


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._num_frames = 0

    def start(self):
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._num_frames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._num_frames / self.elapsed()


def bb_to_rect(x, y, w, h):
    # take a bounding box predicted by face detector and convert it
    # to the format (left, top, right, bottom) as we would normally do
    # with dlib
    left = x
    top = y
    right = w + x
    bottom = h + y

    return (left, top, right, bottom)


def draw_landmarks(frame, faces, lm_predictor):
    (left, top, right, bottom) = bb_to_rect(faces[0], faces[1],
                                            faces[2], faces[3])
    new_rect = dlib.rectangle(int(left),
                              int(top),
                              int(right),
                              int(bottom))
    # For every face rectangle, run landmarkDetector
    landmarks = lm_predictor(frame, new_rect)
    landmarks_points = landmarks.parts()

    for i in range(len(landmarks_points)):
        cv2.circle(frame, (landmarks_points[i].x, landmarks_points[i].y), 1,
                   COLOR_YELLOW,
                   -1)


def write_landmarks_to_file(landmarks, landmarks_fn):
    with open(landmarks_fn, 'w') as f:
        for p in landmarks.parts():
            f.write("%s %s\n" % (int(p.x), int(p.y)))

    f.close()


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)

    rota_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rota_matrix, image.shape[1::-1],
                            cv2.INTER_LINEAR)
    return result


def int_over(number):
    if int(number) < 0:
        return abs(0 - int(number))
    else:
        return 0


def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[0] + rect[2]:
        return False
    elif point[1] > rect[1] + rect[3]:
        return False
    return True
