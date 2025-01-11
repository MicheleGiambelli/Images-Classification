# -*- coding: utf-8 -*-
"""
Author: Michele Giambelli

CNN Architecture for the Image Classification Project

This file defines the architecture of a customized Convolutional Neural Network (CNN) implemented with PyTorch.

Class:
- `MyCNN`: A configurable CNN class that supports multiple convolutional layers, batch normalization, dropout, and customizable activation functions.
"""

import torch
import torch.nn as nn

class MyCNN(nn.Module):
    """
    A custom CNN class for image classification tasks.

    Args:
        input_channels (int): Number of input channels in the images (e.g., 3 for RGB images).
        hidden_layers (list): A list of integers where each element defines the number of filters in a convolutional layer.
        use_batch_normalization (bool): Whether to use batch normalization after each convolutional layer.
        n_output (int): Number of output classes for classification.
        kernel_size (list): A list of kernel sizes for each convolutional layer.
        activation_function (torch.nn.Module): Activation function to apply after each convolution. Default is ReLU.
    """
    def __init__(self, input_channels: int, 
                 hidden_layers: list,
                 use_batch_normalization: bool,
                 n_output: int, 
                 kernel_size: list, 
                 activation_function: torch.nn.Module = torch.nn.ReLU()):
        
        super().__init__()
        
        self.hidden_layers = hidden_layers
        self.use_batch_normalization = use_batch_normalization
        self.activation_function = activation_function
        
        #Build the CNN 
        self.conv_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList() if use_batch_normalization else None
        self.dropout = nn.Dropout(p=0.5)
        
        for i in range(len(hidden_layers)):
            # Add a CNN layer
            layer = nn.Conv2d(in_channels = input_channels, 
                              out_channels = hidden_layers[i], 
                              kernel_size = kernel_size[i],
                              stride=1,
                              padding = kernel_size[i]//2, 
                              padding_mode="zeros",
                              bias = not use_batch_normalization
                              )
            self.conv_layers.append(layer)
                
            if use_batch_normalization:
                self.batch_norm_layers.append(nn.BatchNorm2d(hidden_layers[i]))
        
            input_channels = hidden_layers[i]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.output_layer = nn.Linear(hidden_layers[-1] * (100 // (2**len(hidden_layers)))**2, n_output)
        
    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        
        for i in range(len(self.hidden_layers)):
            input_images = self.conv_layers[i](input_images)
            if self.use_batch_normalization:
                 input_images = self.batch_norm_layers[i](input_images)
            input_images = self.activation_function(input_images)
            input_images = self.pool(input_images)
            input_images = self.dropout(input_images)
        
        input_images = input_images.view(input_images.size(0), -1) #Flatten the tensor
        prediction = self.output_layer(input_images)
        
        return prediction     