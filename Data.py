import gzip
from matplotlib import pyplot as plt
import numpy as np

IMG_DIM = 28


def decode_image_file(fname):
    result = []
    n_bytes_per_img = IMG_DIM * IMG_DIM

    with gzip.open(fname, 'rb') as f:
        bytes_ = f.read()
        data = bytes_[16:]

        if len(data) % n_bytes_per_img != 0:
            raise Exception('Something wrong with the file')

        result = np.frombuffer(data, dtype=np.uint8).reshape(
            len(bytes_) // n_bytes_per_img, n_bytes_per_img)

    return result


def decode_label_file(fname):
    result = []

    with gzip.open(fname, 'rb') as f:
        bytes_ = f.read()
        data = bytes_[8:]

        result = np.frombuffer(data, dtype=np.uint8)

    return result


def normalize(data):
    return data / 255


train_images = decode_image_file('train-images-idx3-ubyte.gz')
train_labels = decode_label_file('train-labels-idx1-ubyte.gz')
data = normalize(train_images)
outputs = train_labels
