import pickle
import torch
import numpy as np

def load_cifar100_file(filename):
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')

    return data_dict

def reshape_image(image_flatten):
    image_shaped = image_flatten.reshape(3, 32, 32)

    return image.astype(np.uint8)

