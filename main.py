# -*- coding: utf-8 -*-
"""
Author: Michele Giambelli

Main file of the Image Classification Project

This file contains the primary functions and logic for training and evaluating a customized CNN for image classification tasks. It includes:
- Dataset preparation and splitting into training, validation, and test sets.
- A training loop with early stopping and performance tracking.
- Model evaluation on the test set.

Functions:
- `evaluate_model`: Evaluates the model on the test dataset and computes accuracy.
- `training_loop`: Handles the training process, including early stopping and performance tracking.
- `main`:  dataset preparation, model initialization, training, and evaluation.
"""

import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import ImagesDataset
from architecture import MyCNN
from utils import accuracy, TransformedImagesDataset


def evaluate_model(model, test_loader, device):
    """Evaluate the model on the test dataset.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to perform the evaluation (CPU or GPU).

    Returns:
        float: Accuracy of the model on the test dataset.
    """
    model.eval() # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y, _, _ in test_loader:         
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    acc= correct/total
    return acc


def training_loop(network: torch.nn.Module, 
                  training_loader, 
                  validation_loader,
                  device,
                  num_epochs: int, 
                  learning_rate = 1e-3, 
                  show_progress: bool = False,
                  patience = 7,
                  ):
    """Training loop function.

    Args:
        network (torch.nn.Module): The neural network model to train.
        training_loader (DataLoader): DataLoader for the training dataset.
        validation_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to perform the training (CPU or GPU).
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        show_progress (bool): Whether to show progress during training.
        patience (int): Number of epochs to wait before early stopping.

    Returns:
        tuple: Lists of training losses, validation losses, training accuracies, and validation accuracies.

    This train loop use a Adam optimizer and a Cross entropy loss function.
    After an epoch of training there is a full iteration on the validation data.
    There is a early stopping implementation that stops the training if after a fixed number of epochs there is no
    improvement in the validation loss function.
    The model with the minimum loss function is stored and at the end of the all training process.
    """
    
    # Create an optimizer Adam
    optimizer = torch.optim.Adam(network.parameters(), lr = learning_rate)
    #Create loss function Cross Entropy
    loss_function = torch.nn.CrossEntropyLoss()

    # Lists to store the training and evaluation losses/accuracy
    train_losses = []
    eval_losses = []
    train_acc = []
    eval_acc = []
    best_eval_loss = float('inf')
    best_eval_acc = 0.0
    best_epoch = 0
    
    # Perform num_epoch full iterations over train_data
    for epoch in range(num_epochs):
        # Training phase
        network.train()
        epoch_train_loss = []
        epoch_train_acc = []

        for input_img, _, _, target, _, _  in tqdm(training_loader, desc = f"Epoch {epoch+1}/{num_epochs}", disable = not show_progress):
            input_img = input_img.to(device)
            target = target.to(device)
            # Reset the gradient
            optimizer.zero_grad()  
            # Compute the output
            output = network(input_img)
            # Compute the loss with Cross-Entropy for the current batch
            loss = loss_function(output, target) 
            # Compute the gradient
            loss.backward()
            # Perform the update
            optimizer.step()
                  
            # Collect minibatch training loss to compute the average loss of the epoch
            epoch_train_loss.append(loss.item())
            #Compute training accuracy
            acc = accuracy(output, target)
            epoch_train_acc.append(acc)
            

        # Average loss for the current epoch
        average_loss_train = torch.mean(torch.tensor(epoch_train_loss))
        train_losses.append(average_loss_train)
        #Average accuracy for the current epoch
        average_train_accuracy = torch.mean(torch.tensor(epoch_train_acc))
        train_acc.append(average_train_accuracy)
        
        # After an epoch of training --> full iteration over evaluation data
        network.eval() # setting network in evaluation mode
        epoch_eval_loss = []
        epoch_eval_acc = []
        
        with torch.no_grad(): # Set no update for the weights
            for input_img, target, _, _ in tqdm(validation_loader, desc = f"Epoch {epoch+1}/{num_epochs}", disable = not show_progress):
                input_img = input_img.to(device)
                target = target.to(device)
                output = network(input_img)
                loss = loss_function(output, target) 
                epoch_eval_loss.append(loss.item())
                acc = accuracy(output, target)
                epoch_eval_acc.append(acc)
        
        average_loss_eval = torch.mean(torch.tensor(epoch_eval_loss))
        eval_losses.append(average_loss_eval)
        average_eval_accuracy = torch.mean(torch.tensor(epoch_eval_acc))
        eval_acc.append(average_eval_accuracy)


        if show_progress:
            print(f"EPOCH: {epoch+1} --- Train loss: {average_loss_train:7.4f} --- Eval loss: {average_loss_eval:7.4f}\n")
            print(f"Train accuracy: {average_train_accuracy:7.4f} --- Eval accuracy: {average_eval_accuracy:7.4f}\n")
            
        # Early stopping
        if average_loss_eval < best_eval_loss:
            best_eval_loss = average_loss_eval
            best_eval_acc = average_eval_accuracy
            best_epoch = epoch
            patience_counter = 0
            torch.save(network.state_dict(), "model.pth") #Save the best model till the current epoch
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs\n")
            # Load the best model at the end of training
            network.load_state_dict(torch.load("model.pth"))
            print(f"BEST MODEL at EPOCH: {best_epoch+1} --- Eval loss: {best_eval_loss:7.4f} --- Eval accuracy: {best_eval_acc:7.4f}")
            break
    
    print("--- Training Ended! ---")
    return train_losses, eval_losses, train_acc, eval_acc, best_epoch



def main(dataset_path,
         network_config: None,
         train_num=0.7, valid_num=0.2, test_num=0.1, #Set the proportion of the 3 sets
         batch_size = 32,
         learning_rate: float = 1e-3,
         device: str = "cuda",
         num_epochs = 100,
        ):
    
    """Main function to train and evaluate the model.

    Args:
        dataset_path (str): Path to the dataset.
        network_config (dict): Configuration for the network architecture.
        train_num (float): Proportion of the dataset for training.
        valid_num (float): Proportion of the dataset for validation.
        test_num (float): Proportion of the dataset for testing.
        batch_size (int): Batch size for DataLoader.
        learning_rate (float): Learning rate for the optimizer.
        device (str): Device to use (e.g., "cuda" or "cpu").
        num_epochs (int): Number of training epochs.
    """
    
    device = torch.device(device)
    if "cuda" in device.type and not torch.cuda.is_available():
        warnings.warn("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    
    np.random.seed(0)
    torch.manual_seed(0)
    image_dataset = ImagesDataset(dataset_path)

    #Create training, validation and test set
    #Define the number of element for each set 
    rng = np.random.default_rng(seed=0)
    n_samples = len(image_dataset)
    shuffled_idx = rng.permutation(n_samples)

    train_size = int(n_samples * train_num)
    val_size = int(n_samples * valid_num)
    training_idx = shuffled_idx[:train_size]
    validation_idx = shuffled_idx[train_size:train_size + val_size]
    test_idx = shuffled_idx[train_size + val_size:]

    # Create PyTorch subsets from our subset indices
    training_set = Subset(image_dataset, indices = training_idx)
    validation_set = Subset(image_dataset, indices = validation_idx)
    test_set = Subset(image_dataset, indices = test_idx)

    #Agumented data
    training_set = TransformedImagesDataset(training_set)

    #Creating Data Loaders
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    #TRAINING THE MODEL
    model = MyCNN(input_channels = network_config['input_channels'], 
                  hidden_layers = network_config['n_hidden_layers'], 
                  use_batch_normalization = network_config['use_batch_normalization'],  
                  n_output = network_config['n_outputs'], 
                  kernel_size = network_config['kernel_size'])
    model.to(device)
    
    train_losses, val_losses, train_acc, val_acc, best_epoch = training_loop(model, 
                                                                             training_loader, 
                                                                             validation_loader,
                                                                             device,
                                                                             num_epochs=config['num_epochs'],
                                                                             learning_rate=learning_rate,
                                                                             show_progress=True)
    
    #TEST THE MODEL
    test_accuracy = evaluate_model(model, test_loader, device)
    print(f"Test accuracy: {test_accuracy:7.4f}")



if __name__ == "__main__":
    import json
    dataset_path = input("Please enter the dataset path: ")
    
    with open('working_config.json', 'r') as config_file:
        config = json.load(config_file)
    
    main(dataset_path = dataset_path,
         batch_size = config['batch_size'],
         network_config = config['network_config'],
         learning_rate = config['learning_rate'],
         device = config['device'])