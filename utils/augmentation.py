import torch
import albumentations as A
import cv2

transform_pipeline = A.Compose([
    A.HorizontalFlip(p=.5),
    A.RandomBrightnessContrast(p=.2),
])
