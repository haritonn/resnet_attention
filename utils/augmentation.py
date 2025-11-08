import torch
import albumentations as A
import cv2

# Current augmentation pipeline (can experiment here)
transform_pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ]
)
