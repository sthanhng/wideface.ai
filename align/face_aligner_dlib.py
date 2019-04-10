import cv2
import numpy as np

from misc import shape_to_np
from misc import FACIAL_LANDMARKS_68_IDXS, FACIAL_LANDMARKS_5_IDXS


class FaceAligner:
    """
    Use the dlib's landmarks estimation to align faces

    The alignment preprocess faces for input into a neural network to encode
    the faces.
    Faces are resized to the same size (such as 96x96) and transformed to
    make landmarks (such as the eyes and nose) appear at the same location on every image.

    """

    def __init__(self, predictor, desired_left_eye=(0.35, 0.35),
                 desired_face_width=256, desired_face_height=None):
        """
        Instantiate an 'FaceAligner' object.

        :param predictor: the path to the face detector
        :param desired_left_eye:
        :param desired_face_width:
        :param desired_face_height:
        """

        assert predictor is not None

        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = predictor
        self.desiredLeftEye = desired_left_eye
        self.desiredFaceWidth = desired_face_width
        self.desiredFaceHeight = desired_face_height

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, gray, rect):
        """
        Transform and align a face in an image
        :param image: the image to process
        :param gray: the gray image
        :param rect: bounding box around the face to align
        :return: the aligned image
        """

        assert image is not None

        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)

        if len(shape) == 68:
            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        else:
            (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]

        left_eye_pts = shape[lStart:lEnd]
        right_eye_pts = shape[rStart:rEnd]

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
