import torch 
from torch.utils.data import Dataset
import numpy as np

from utils.utils import load_cifar100_file, reshape_image
from utils.augmentation import transform_pipeline

class CifarDataset(Dataset):
    """
    Class for CIFAR-100 Dataset (from pickels)
    Args:
        data_path: path to current dataset (train/test)
        is_train: is current dataset for training (for augmentation)
    """
    def __init__(self, data_path, is_train=True):
        super().__init__()
        loaded = load_cifar100_file(data_path)
        images_flatten = loaded[b'data']
        # (50000, 3072) -> (50000, 32, 32, 3)
        self.images = images_flatten.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) 
        self.images = self.images.astype(np.uint8) / 255.0
        self.fine_labels = loaded[b'fine_labels']
        self.is_train = is_train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        labels_idx = self.fine_labels[idx]
        images_idx = self.images[idx]


        if self.is_train:
            images_idx = transform_pipeline(image=images_idx)['image']
        images_idx = images_idx.transpose(2, 0, 1)

        images = torch.tensor(images_idx, dtype=torch.float32)
        labels = torch.tensor(labels_idx, dtype=torch.long)

        return images, labels
