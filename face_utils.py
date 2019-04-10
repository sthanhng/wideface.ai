import numpy as np
import tensorflow as tf
import os
import pickle
import re

from tensorflow.python.platform import gfile
from misc import to_rgb, prewhiten
from misc import crop, flip

from scipy import misc


def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding
    and get a euclidean distance for each comparison face.
    The distance tells you how similar the faces are.

    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order
    as the 'faces' array
    """

    if len(face_encodings) == 0:
        return np.empty(0)

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """
    Compare a list of face encodings against a candidate encoding to see
    if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding
    to compare against the list
    :param tolerance: How much distance between faces to consider it a match.
    Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings
    match the face encoding to check
    """

    return list(face_distance(known_face_encodings, face_encoding_to_check)
                <= tolerance)


def create_face_embeddings(model, data_dir, image_size, batch_size):
    with tf.Graph().as_default():
        with tf.Session() as sess:

            # Get the paths for the corresponding images
            flexface = [data_dir + f for f in os.listdir(data_dir)]
            paths = flexface
            # np.save("images.npy",paths)
            # Load the model
            load_model(model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name(
                "input:0")
            images_placeholder = tf.image.resize_images(images_placeholder,
                                                        (160, 160))
            embeddings = tf.get_default_graph().get_tensor_by_name(
                "embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name(
                "phase_train:0")

            image_size = image_size
            embedding_size = embeddings.get_shape()[1]
            print("Embedding size: {}".format(embedding_size))
            extracted_dict = {}

            # Run forward pass to calculate embeddings
            for i, filename in enumerate(paths):
                images = load_image(filename, False, False, image_size)
                feed_dict = {images_placeholder: images,
                             phase_train_placeholder: False}
                feature_vector = sess.run(embeddings, feed_dict=feed_dict)
                extracted_dict[filename] = feature_vector
                if i % batch_size == 0:
                    print("completed", i, " images")

            with open('extracted_dict.pickle', 'wb') as f:
                pickle.dump(extracted_dict, f)


def load_model(model):
    # Check if the model is a model directory (containing a metagraph and
    # a checkpoint file) or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(),
                      os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError(
            'No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError(
            'There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]

    return meta_file, ckpt_file


def load_image(img, do_random_crop, do_random_flip, image_size,
               do_prewhiten=True):
    # nrof_samples = len(image_paths)
    images = np.zeros((1, image_size, image_size, 3))
    # for i in range(nrof_samples):
    img = misc.imread(img)
    # img = misc.imresize(img,(160,160,3))
    if img.ndim == 2:
        img = to_rgb(img)
    if do_prewhiten:
        img = prewhiten(img)
    img = crop(img, do_random_crop, image_size)
    img = flip(img, do_random_flip)
    images[:, :, :, :] = img

    return images
