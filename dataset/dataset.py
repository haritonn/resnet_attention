import torch 
from torch.utils.data import Dataset

from utils import load_cifar100_file, reshape_image
from augmentation import transform_pipeline

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
        self.images = reshape_image(images_flatten)
        self.fine_labels = loaded[b'fine_labels']
        self.is_train = is_train


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        labels_idx = self.fine_labels[idx]
        images_idx = self.images[idx]
        if self.is_train:
            images_idx = self.transform_pipeline(image=images_idx)['image']

        images = torch.tensor(images_idx, dtype=torch.float32)
        labels = torch.tensor(labels_idx, dtype=torch.long)

        return images, labels
