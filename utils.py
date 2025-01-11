# -*- coding: utf-8 -*-
"""
Author: Michele Giambelli

Utility file for Image Classification Project

This file contains utility functions and classes used for:
- Data augmentation during training.
- Accuracy evaluation of the model during training and validation.

Functions and Classes:
- `augment_image`: Applies various transformations to an image for augmentation.
- `TransformedImagesDataset`: A custom PyTorch Dataset class for handling augmented images.
- `accuracy`: Computes accuracy for predictions against true labels.
"""
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import random

def augment_image(img, index: int):
    """Applies a specific augmentation to the input image based on the index.

    Args:
        img (PIL.Image): The input image.
        index (int): An index determining which augmentation to apply.

    Returns:
        Tuple: Augmented image and a string describing the applied transformation.
    """

    transformations_list = [
        transforms.GaussianBlur(kernel_size = (5, 9), sigma=(0.1, 3)),
        transforms.RandomRotation(degrees=90),  #degrees=Range of degrees to select from
        transforms.RandomVerticalFlip(), #set p=1 to have 100% of probability of vertical flip 
        transforms.RandomHorizontalFlip(), #set p=1 if we want it
        transforms.ColorJitter(brightness=0.5, hue=0.3)
    ]

    v = index % 7
        
    if v == 0:
        transformed_image = img
        return transformed_image, "Original"
    
    if v == 1:
        transformed_image = transformations_list[0](img)
        return transformed_image, "GaussianBlur"
    
    if v == 2:
        transformed_image = transformations_list[1](img)
        return transformed_image, "RandomRotation"
    
    if v == 3:
        transformed_image = transformations_list[2](img)
        return transformed_image, "RandomVerticalFlip"
    
    if v == 4:
        transformed_image = transformations_list[3](img)
        return transformed_image, "RandomHorizontalFlip"
    
    if v == 5:
        transformed_image = transformations_list[4](img)
        return transformed_image, "ColorJitter"        
    
    if v == 6:
        random_transforms = random.sample(transformations_list, 3)
        transformation = transforms.Compose([
            random_transforms[0],
            random_transforms[1],
            random_transforms[2]
        ])
        transformed_image = transformation(img)
        return transformed_image, "Compose"
    

class TransformedImagesDataset(Dataset):
    """Custom Dataset class for handling image data with augmentations.

    Args:
        data_set (Dataset): The original dataset to augment.
    """
    def __init__(self, data_set: Dataset):
        super().__init__()
        self.data_set = data_set
    
    def __getitem__(self, index: int):
        img, class_ID, class_name, img_path = self.data_set[index // 7]
        transform_img, transform_name = augment_image(img, index)
        
        return transform_img, transform_name, index, class_ID, class_name, img_path
    
    def __len__(self):
        return len(self.data_set)*7


def accuracy(predictions, targets):
    """Calculates accuracy of predictions compared to targets.

    Args:
        predictions (torch.Tensor): The model's predictions.
        targets (torch.Tensor): The true labels.

    Returns:
        float: The accuracy as a proportion of correct predictions.
    """
    correct = 0
    total = 0 
    _, pred_labels = torch.max(predictions, 1)
    total = targets.size(0)
    correct = (pred_labels == targets).sum().item()
    
    accuracy = correct / total
    return accuracy