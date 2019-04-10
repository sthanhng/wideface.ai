import cv2
import numpy as np

from misc import shape_to_np
from misc import FACIAL_LANDMARKS_68_IDXS
from face_api import FaceAPI
from utils.colors import ALICEBLUE


class FaceAligner:
    """
    Use the dlib's or caffe's landmarks estimation to align faces

    The alignment preprocess faces for input into a neural network to encode
    the faces.
    Faces are resized to the same size (such as 96x96) and transformed to
    make landmarks (such as the eyes and nose) appear at the same location on every image.

    """

    def __init__(self, model, predictor, desired_left_eye=(0.35, 0.35),
                 desired_face_width=160, desired_face_height=None):
        """
        Instantiate an 'FaceAligner' object.

        :param predictor: the path to the face detector. Here we support two
        face detectors, include dlib's face detector and OpenCV DNN face
        detector using caffe/tensorflow models
        :param desired_left_eye:
        :param desired_face_width:
        :param desired_face_height:
        """

        assert model, predictor is not None

        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.model = model
        self.predictor = predictor
        self.desiredLeftEye = desired_left_eye
        self.desiredFaceWidth = desired_face_width
        self.desiredFaceHeight = desired_face_height

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, bounding_box, draw=False, color=None):
        """
        Transform and align a face in an image
        :param image: the image to process
        :param bounding_box: bounding box around the face to align
        :param color: color of the landmark points
        :param draw: set to display or don't diplay the eye's landmark points on
        the image
        :return: the aligned image
        """

        assert image is not None

        # Using dlib's landmarks estimation or caffe model
        if self.model == 'dlib':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # convert the landmark (x, y)-coordinates to a NumPy array
            shape = self.predictor(gray, bounding_box)
            shape = shape_to_np(shape)
        else:
            LM_PARAM = 60
            list_caffe_lm = FaceAPI.caffe_find_landmarks(self, self.predictor,
                                                         image,
                                                         bounding_box,
                                                         LM_PARAM, False,
                                                         ALICEBLUE)

        # extract the left and right eye (x, y)-coordinates
        (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]

        if self.model == 'dlib':
            left_eye_pts = shape[lStart:lEnd]
            right_eye_pts = shape[rStart:rEnd]
        else:
            left_eye_pts = np.asarray(list_caffe_lm[0][lStart:lEnd])
            right_eye_pts = np.asarray(list_caffe_lm[0][rStart:rEnd])

        # Draw the landmark points on the face detected
        if draw:
            for lp in left_eye_pts:
                cv2.circle(image, tuple(lp), 1, color, 2)
            for rp in right_eye_pts:
                cv2.circle(image, tuple(rp), 1, color, 2)
        else:
            print("Eye's Landmark points don't display.")

        # compute the center of mass for each eye
        left_eye_center = left_eye_pts.mean(axis=0).astype("int")
        right_eye_center = right_eye_pts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desired_right_eye_x = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desired_dist = (desired_right_eye_x - self.desiredLeftEye[0])
        desired_dist *= self.desiredFaceWidth
        scale = desired_dist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                       (left_eye_center[1] + right_eye_center[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        aligned_face = cv2.warpAffine(image, M, (w, h),
                                      flags=cv2.INTER_CUBIC)

        # return the aligned face
        return aligned_face
