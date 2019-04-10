import cv2
import os
import dlib
import numpy as np
import time
import datetime as dt

from utils.colors import *
from misc import rect_to_bb, shape_to_np
from misc import custom_draw_bb, exists_directory, write_face_data_json_file
from utils.yoloface import get_outputs_names, post_process


class FaceAPI:
    """

    """

    def __init__(self, config, face_detector, landmarks_predictor,
                 face_aligner=None):

        self.config = config
        self.face_detector = face_detector
        self.landmarks_predictor = landmarks_predictor
        self.face_aligner = face_aligner

    def detect_face(self, detector, image):
        """
        Detect the faces in an image using the DNN model of OpenCV
        :param detector: the DNN model. Here we use 2 networks:
            - FP16 version of the original caffe implementation (5.4 MB)
            - or 8 bit Quantized version using Tensorflow (2.7 MB)
        :param image: the frame captured from the camera/webcam
        :return: frame and the list of faces detected
        """

        frame = image.copy()
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300),
                                     [104, 117, 123], False, False)
        detector.setInput(blob)
        detections = detector.forward()

        bboxes = []
        list_confidence = []
        list_refined_box = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.config["face_detect"]["conf_threshold"]:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)

                left, top, right, bottom = x1, y1, x2, y2
                original_vertical_length = bottom - top
                top = int(top + original_vertical_length * 0.15)
                bottom = int(bottom - original_vertical_length * 0.05)

                margin = ((bottom - top) - (right - left)) // 2
                left = left - margin if (bottom - top - right + left) % 2 == \
                                        0 else left - margin - 1
                right = right + margin

                bboxes.append([x1, y1, x2, y2])
                list_confidence.append(confidence)
                refined_box = [left, top, right, bottom]
                list_refined_box.append(refined_box)
                custom_draw_bb(frame, left, top, (right - left), (bottom - top),
                               YELLOW1, 2)

        return frame, bboxes, list_refined_box

    def dlib_detect_face(self, detector, image,
                         number_of_times_to_upsample=1, draw=False, color=None):
        """

        :param detector:
        :param image:
        :param number_of_times_to_upsample:
        :param draw:
        :param color:
        :return:
        """
        dlib_image = image.copy()
        gray = cv2.cvtColor(dlib_image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, number_of_times_to_upsample)
        # Draw the rectangle around the faces detected
        if draw:
            for rect in rects:
                (bX, bY, bW, bH) = rect_to_bb(rect)
                custom_draw_bb(dlib_image, bX, bY, bW, bH, color, 2)

        return dlib_image, rects

    def crop_face(self, frame, face, face_idx, file_name, total_face=0):
        """
        Crop the face detected in an image
        :param frame: captured frame
        :param face: list of faces detected
        :param face_idx: index of the face
        :param file_name: name of the cropped image will be saved
        :param total_face: number of the faces detected
        :return: cropped image
        """

        self.cropped_face = frame[face[1]:face[3],
                            face[0]:face[2]]

        p_cropped_face = os.path.sep.join(
            [self.config["dataset"]["cropped_faces"], file_name])
        cv2.imwrite(p_cropped_face, self.cropped_face)
        total_face += 1
        print("{} faces saved at {}".format(total_face, p_cropped_face))

        box = {}
        box['left'] = face[0]
        box['top'] = face[1]
        box['right'] = face[2]
        box['bottom'] = face[3]
        box_as_list = {}
        box_as_list['box'] = box

        return total_face, box_as_list

    def dlib_find_landmarks(self, predictor, image, bounding_box, draw=False,
                            color=None):
        """

        :param predictor:
        :param image:
        :param bounding_box:
        :param draw:
        :param color:
        :return:
        """

        image_dlib = image.copy()
        for rect in bounding_box:
            # ----------------------------------------------------
            #
            # For every face bounding-box, run landmark detector
            #
            # ----------------------------------------------------
            landmarks = predictor(image, rect)
            points = landmarks.parts()

            if draw:
                # Loop over the (x, y) coordinates for the facial landmarks
                # and draw them on the image
                for p in points:
                    cv2.circle(image_dlib, (p.x, p.y), 1, color, 2)

        return image_dlib

    def yolo_face_detect(self, net, frame, img_width, img_height):
        """

        :param net:
        :param frame:
        :return:
        """

        frame_yolo = frame.copy()

        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(cv2.resize(frame_yolo, (300, 300)),
                                     1 / 255, (img_width, img_height),
                                     [0, 0, 0], 1, crop=False)
        # Sets the input to the network
        net.setInput(blob)
        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))

        # Remove the bounding boxes with low confidence
        rects = post_process(frame_yolo, outs, 0.5, 0.4)
        list_dlib_rect = []
        for i in range(len(rects)):
            original_vertical_length = rects[i][3]
            top = int(rects[i][1] + original_vertical_length * 0.15)
            bottom = int(rects[i][3] - rects[i][1] - original_vertical_length *
                         0.02)

            margin = ((bottom - top) - rects[i][2]) // 2
            left = rects[i][0] - margin if (bottom - top - rects[i][2]) % 2 == \
                                           0 else rects[i][0] - margin + 1
            right = (rects[i][2] - left) + margin

            rect_bb = dlib.rectangle(left=left, top=top, right=right,
                                     bottom=bottom)
            list_dlib_rect.append(rect_bb)

        return frame_yolo, rects, list_dlib_rect

    def caffe_find_landmarks(self, predictor, image, bounding_box, lm_param,
                             draw=False, color=None):
        """

        :param predictor:
        :param image:
        :param bounding_box:
        :param lm_param:
        :param draw:
        :param color:
        :return:
        """

        left, top, right, bottom = bounding_box
        roi = image[top:bottom + 1, left:right + 1]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized_roi = cv2.resize(gray_roi, (lm_param, lm_param)).astype(
            np.float32)

        m = np.zeros((lm_param, lm_param))
        sd = np.zeros((60, 60))
        mean, std_dev = cv2.meanStdDev(resized_roi, m, sd)
        normalized_roi = (resized_roi - mean[0][0]) / (0.000001 + std_dev[0][0])

        # ----------------------------------------------------
        #
        # For every face bounding-box, run landmark detector
        #
        # ----------------------------------------------------
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(normalized_roi, 1.0, (lm_param, lm_param),
                                     None)
        # Sets the input to the network
        predictor.setInput(blob)
        caffe_landmarks = predictor.forward()

        list_caffe_lm = []
        for landmark in caffe_landmarks:
            list_lm = []
            for i in range(len(landmark) // 2):
                x = landmark[2 * i] * (right - left) + left
                y = landmark[2 * i + 1] * (bottom - top) + top
                list_lm.append((int(x), int(y)))
            list_caffe_lm.append(list_lm)

        # Draw the landmark points on the image
        if draw:
            for lm in list_caffe_lm:
                for idx, p in enumerate(lm):
                    cv2.circle(image, p, 1, color, 2)

        # Return the list of landmark points
        return list_caffe_lm

    def caffe_find_landmarks_multi_face(self, predictor, image, bounding_box,
                                        lm_param, draw=False, color=None):
        """

        :param predictor:
        :param image:
        :param bounding_box:
        :param lm_param:
        :param draw:
        :param color:
        :return:
        """

        for rect in bounding_box:
            self.caffe_find_landmarks(predictor, image, rect, lm_param, draw,
                                      color)

    def get_dataset_from_webcam(self, source=0, save_video=False,
                                landmarks=False):
        # The dataset directories
        jpeg_images = self.config["dataset"]["jpeg_images"]
        cropped_faces = self.config["dataset"]["cropped_faces"]
        aligned_faces = self.config["dataset"]["aligned_faces"]
        json_dir = self.config["dataset"]["json"]
        video_dir = self.config["dataset"]["videos"]

        exists_directory(jpeg_images)
        exists_directory(cropped_faces)
        exists_directory(json_dir)
        exists_directory(aligned_faces)
        exists_directory(video_dir)

        vs = cv2.VideoCapture(source)
        ret, frame = vs.read()
        frame_count, tt, total_image, total_face, time_shot = 0, 0, 0, 0, 0
        time_video_record = dt.datetime.now()

        # Video writer
        video_writer = cv2.VideoWriter(
            os.path.join(video_dir,
                         '{}-{}.avi'.format(time_video_record.strftime(
                             '%Y-%m-%d_%H-%M-%S'),
                             str(source).split(".")[0])),
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15,
            (frame.shape[1], frame.shape[0]))

        while True:
            ret, frame = vs.read()
            orig_image = frame.copy()
            if not ret:
                break
            frame_count += 1

            start_time = time.time()
            outs, bboxes, list_refined_box = self.detect_face(
                self.face_detector, frame)
            tt += time.time() - start_time
            fps = frame_count / tt
            text = "FLEX FACE | {:.1f} FPS".format(fps)
            cv2.putText(outs, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        CYAN2, 1, cv2.LINE_AA)
            # Get the date and time at current time
            time_now = dt.datetime.now()
            cv2.putText(outs, time_now.strftime('%Y-%m-%d %H:%M:%S'), (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN2, 1, cv2.LINE_AA)
            time_save = dt.datetime.now()
            base_file_name = '{}'.format(
                time_save.strftime('%Y-%m-%d_%H-%M-%S') + '_' + str(
                    total_image).zfill(6))
            file_name_img = base_file_name + '.jpg'

            if len(bboxes) > 0:
                # ---------------------------------------------------------
                #
                # Draw the landmark points on the image
                #
                # ---------------------------------------------------------
                if landmarks:
                    # For every face detected, draw the rectangle around the
                    # face and save the image
                    self.caffe_find_landmarks_multi_face(
                        self.landmarks_predictor, frame, bboxes, 60, True,
                        YELLOW1)

                if time.time() - time_shot > self.config["face_detect"][
                    "capture_dur"]:
                    p = os.path.sep.join([jpeg_images, file_name_img])
                    cv2.imwrite(p, orig_image)
                    total_image += 1
                    time_shot = time.time()
                    print("{} frames saved as {}".format(total_image, p))

                    list_face = []
                    # ---------------------------------------------------------
                    #
                    # Save copped and aligned images
                    #
                    # ---------------------------------------------------------
                    for (i, rect) in enumerate(bboxes):
                        face_orig = frame[rect[1]:rect[3], rect[0]:rect[2]]
                        face_aligned = self.face_aligner.align(frame, rect)
                        cv2.imshow("Original", face_orig)
                        cv2.imshow("Aligned", face_aligned)

                        cropped_fn = '{}.jpg'.format(
                            time_save.strftime('%Y-%m-%d_%H-%M-%S') +
                            '_cropped_' + str(total_face).zfill(6))
                        aligned_fn = '{}.jpg'.format(
                            time_save.strftime('%Y-%m-%d_%H-%M-%S') +
                            '_aligned_' + str(total_face).zfill(6))
                        total_face, box = self.crop_face(orig_image, rect, i,
                                                         cropped_fn, total_face)
                        cv2.imwrite(os.path.join(aligned_faces, aligned_fn),
                                    face_aligned)
                        list_face.append(box)

                    # ---------------------------------------------------------
                    #
                    # Write faces detected to JSON file
                    #
                    # ---------------------------------------------------------
                    face_data = [orig_image.shape[0], orig_image.shape[1],
                                 len(bboxes), file_name_img, list_face]
                    write_face_data_json_file(json_dir, base_file_name,
                                              face_data)
            if frame_count == 1:
                tt = 0

            cv2.imshow("Flex face", outs)
            if save_video:
                video_writer.write(outs)
            k = cv2.waitKey(1)
            if k == 27:
                break
        cv2.destroyAllWindows()
        video_writer.release()
