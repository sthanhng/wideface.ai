import cv2
import numpy as np

from misc import FACIAL_LANDMARKS_68_IDXS


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

    def align(self, image, rect):
        """
        Transform and align a face in an image
        :param image: the image to process
        :param gray: the gray image
        :param rect: bounding box around the face to align
        :return: the aligned image
        """

        assert image is not None

        # convert the landmark (x, y)-coordinates to a NumPy array

        left, top, right, bottom = rect
        roi = image[top:bottom + 1, left:right + 1]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        res = cv2.resize(gray_roi, (60, 60)).astype(np.float32)

        m = np.zeros((60, 60))
        std_dev = np.zeros((60, 60))
        mean, std_dev = cv2.meanStdDev(res, m, std_dev)
        normalized_roi = (res - mean[0][0]) / (0.000001 + std_dev[0][0])

        # For every face rectangle, run landmarkDetector
        blob = cv2.dnn.blobFromImage(normalized_roi, 1.0, (60, 60), None)
        self.predictor.setInput(blob)
        caffe_landmarks = self.predictor.forward()

        list_CLM = []
        for landmark in caffe_landmarks:
            LM = []
            for i in range(len(landmark) // 2):
                x = landmark[2 * i] * (right - left) + left
                y = landmark[2 * i + 1] * (bottom - top) + top
                LM.append((int(x), int(y)))
            list_CLM.append(LM)

        for lm in list_CLM:
            for idx, p in enumerate(lm):
                cv2.circle(image, p, 1, (0, 0, 255), 2)


        # extract the left and right eye (x, y)-coordinates
        (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]

        left_eye_pts = list_CLM[0][lStart:lEnd]
        right_eye_pts = list_CLM[0][rStart:rEnd]

        for p in left_eye_pts:
            cv2.circle(image, (p[0], p[1]), 1, (255, 0, 0), 2)

        for p in right_eye_pts:
            cv2.circle(image, (p[0], p[1]), 1, (255, 0, 0), 2)

        left_eye_pts = np.asarray(left_eye_pts)
        right_eye_pts = np.asarray(right_eye_pts)

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
