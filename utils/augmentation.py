import torch
import albumentations as A
import cv2

transform_pipeline = A.Compose([
    A.HorizontalFlip(p=.5),
    A.RandomRotate90(p=.5),
    A.RandomBrightnessContrast(p=.3),
    A.RandomFog(p=.4),
    A.Normalize(mean=(.485, .486, .406), std=(.229, .224, .225)),
])
