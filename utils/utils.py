import pickle
import copy
import torch
import numpy as np

class EarlyStopping:
    """
    Class for early stopping implementation
    """
    def __init__(self, patience=5, epsilon=.005):
        self.patience = patience
        self.epsilon = epsilon

        self.best_score = 100
        self.curr_patience = 0
        self.curr_model_state = None

    def step(self, curr_score, model):
        if curr_score >= self.best_score - self.epsilon:
            self.curr_patience += 1
            if self.curr_patience >= self.patience:
                return False, self.curr_model_state
        else:
            self.curr_patience = 0
            self.best_score = curr_score
            self.curr_model_state = copy.deepcopy(model.state_dict())
            return True, None


def load_cifar100_file(filename):
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')

    return data_dict

def reshape_image(image_flatten):
    image_shaped = image_flatten.reshape(3, 32, 32)

    return image.astype(np.uint8) / float(255) #[0, 255] -> [0, 1]

