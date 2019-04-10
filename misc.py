import os
import json
import numpy as np
import cv2

from collections import OrderedDict

## Helper functions


# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions

# For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

# For dlib’s 5-point facial landmark detector:
FACIAL_LANDMARKS_5_IDXS = OrderedDict([
    ("right_eye", (2, 3)),
    ("left_eye", (0, 1)),
    ("nose", (4))
])

# in order to support legacy code, we'll default the indexes to the
# 68-point model
FACIAL_LANDMARKS_IDXS = FACIAL_LANDMARKS_68_IDXS


def write_face_data_json_file(path, file_name, face_data):
    """
    Write information of the captured images and faces detected to JSON file.
    The infor include the following fields:
        - width
        - height
        - number of faces in the image
        - name of the image
        - bounding-box of each the face

    :param path: path to the JSON file will be saved
    :param file_name: the name of the JSON file (based on the base name of
    the image captured)
    :param face_data: the information of the image and faces. It's a list
    which has the structure as below:
        ['width', 'height',
        'number of the faces',
        'file name',
        'faces']
    where, the faces field contains the bouding-box of the faces detected in
    an image.
        bounding-box: {'left', 'top', 'right', 'bottom'}
    :return: a JSON file saved on local disk
    """

    json_fpn = os.path.join(path, file_name + '.json')

    data = {}
    json_data = {}
    data['width'] = face_data[0]
    data['height'] = face_data[1]
    data['#faces'] = face_data[2]
    data['name'] = face_data[3]
    data['faces'] = face_data[4]
    json_data['data'] = data

    with open(json_fpn, 'w') as json_file:
        json_file.write(json.dumps(json_data, indent=2, separators=(',', ': '),
                                   ensure_ascii=False))


def write_cfg_file(config_path, file_name, option):
    """

    :param config_path:
    :param file_name:
    :param option:
    :return:
    """

    path_fn = os.path.join(config_path, file_name + '.json')
    with open(path_fn, 'w') as f:
        f.write(json.dumps(option, indent=2, separators=(',', ': '),
                           ensure_ascii=False))


def load_cfg_file(config_path):
    """

    :param config_path:
    :return:
    """

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    return config


def exists_directory(path_dir):
    """

    :param path_dir:
    :return:
    """

    if not os.path.exists(path_dir):
        print("Creating the directory {}...".format(path_dir))
        os.makedirs(path_dir)
    else:
        print("The directory {} already exists!".format(path_dir))


def shape_to_np(shape, dtype="int"):
    """
    Convert the landmark (x, y)-coordinates to a NumPy array
    :param shape: The list of landmarks (x, y)--coordinates
    :param dtype: type of the landmarks (x, y)--coordinates
    :return: a Numpy array
    """
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return x, y, w, h


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


# Create a black image
# width = 512
# height = 512
# path_img = "C:\\workspace\\facial-landmarks\\images\\"
# img = np.zeros((height,width,3), np.uint8)
# img = cv2.imread("example_05.png")
# height, width = img.shape[:2]


def custom_draw_bb(image, x, y, width, height, color=(0, 255, 0), thickness=3,
                   c=0.3):
    x_init = x
    y_init = y
    # Draw a blue line with thickness of 2 px
    # Top-left
    cv2.line(image, (x_init, y_init), (x_init + int(c * width), y_init), color,
             thickness)
    cv2.line(image, (x_init, y_init), (x_init, y_init + int(c * height)), color,
             thickness)

    # Top-right
    # cv2.line(image, (int((1-c)*width), y_init), (width-x_init, y_init), color, thickness)
    cv2.line(image, (x_init + int((1 - c) * width), y_init),
             (width + x_init, y_init), color, thickness)
    # cv2.line(image, (width-x_init, y_init), (width-x_init, int(c*height)), color, thickness)
    cv2.line(image, (width + x_init, y_init),
             (width + x_init, y_init + int(c * height)), color, thickness)

    # # Bottom-left
    # cv2.line(image, (x_init, int((1-c)*height)), (x_init, height-y_init), color, thickness)
    # cv2.line(image, (x_init, height-y_init), (int(c*width), height-y_init), color, thickness)
    cv2.line(image, (x_init, y_init + int((1 - c) * height)),
             (x_init, height + y_init), color, thickness)
    cv2.line(image, (x_init, height + y_init),
             (x_init + int(c * width), height + y_init), color, thickness)

    # # Bottom-right
    # cv2.line(image, (int((1-c)*width), height-y_init), (width-x_init, height-y_init), color, thickness)
    # cv2.line(image, (width-x_init, height-y_init), (width-x_init, int((1-c)*height)), color, thickness)
    cv2.line(image, (x_init + int((1 - c) * width), height + y_init),
             (width + x_init, height + y_init), color, thickness)
    cv2.line(image, (width + x_init, height + y_init),
             (width + x_init, y_init + int((1 - c) * height)), color, thickness)

# draw_bb(img, width, height, thickness=2, c=0.3)
# #Display the image
# cv2.imshow("Output",img)
# cv2.waitKey(0)


def crop(image, random_crop, image_size):
    if image.shape[1] > image_size:
        sz1 = int(image.shape[1] // 2)
        sz2 = int(image_size // 2)
        if random_crop:
            diff = sz1 - sz2
            (h, v) = (np.random.randint(-diff, diff + 1),
                      np.random.randint(-diff, diff + 1))
        else:
            (h, v) = (0, 0)
        image = image[(sz1 - sz2 + v):(sz1 + sz2 + v),
                (sz1 - sz2 + h):(sz1 + sz2 + h), :]
    return image


def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


#
# if __name__ == "__main__":
#     configs = load_cfg_file("dataset/json/2018-10-24_10-22-32_00000002.json")
#     print(configs)
