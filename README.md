# Utkarsh-ERA-V2-Assignment-7
Assignment 7
# MNIST Digit Classification using Convolutional Neural Networks (CNNs)

This project explores the use of Convolutional Neural Networks (CNNs) for digit classification on the MNIST dataset. We experiment with different model architectures and hyperparameters to achieve high accuracy in classifying handwritten digits.

## Project Structure

The project is organized into three main modules:

1. **data**: This module handles the data loading and preprocessing. It includes the `data.py` file, which defines the `get_dataloaders` function. This function loads the MNIST dataset, applies transformations, and returns the data loaders for training and testing.

2. **models**: This module contains the implementation of different CNN architectures. Each model is defined in a separate file (e.g., `Model_1.py`, `Model_2.py`, etc.) and follows a similar structure. The models are implemented using the PyTorch framework and vary in terms of the number and size of convolutional layers, pooling layers, and fully connected layers.

3. **main**: The main module consists of the `main.ipynb` file, which is a Jupyter Notebook that brings together the data and model modules. It handles the training and evaluation of the models. The notebook loads the data, defines the model architecture, sets up the optimizer and loss function, and performs the training and testing loops. It also includes code for visualizing the results and saving the trained models.

## Dataset

The MNIST dataset is a widely-used benchmark dataset for image classification tasks. It consists of 70,000 grayscale images of handwritten digits, with each image being 28x28 pixels in size. The dataset is split into 60,000 training images and 10,000 test images.

## Model Architectures

In this project, we experiment with 8 different CNN architectures to explore their performance on the MNIST dataset. The models are implemented using the PyTorch framework and vary in terms of the number and size of convolutional layers, pooling layers, and fully connected layers.

### Model 1

#### Target:

Get the basic skeleton right. We will try and avoid changing this skeleton as much as possible.
No fancy stuff
#### Results:
Parameters: 6,379,786
Best Train Accuracy: 100.00
Best Test Accuracy: 99.21
#### Analysis:
The model is quite large, but it is working well.
We observe some slight overfitting, as the train accuracy reaches 100% while the test accuracy is slightly lower.

| Layer             | Kernel Size | Stride | Padding | Receptive Field |
|-------------------|-------------|--------|---------|-----------------|
| Input             | -           | -      | -       | 1x1             |
| Layer 1 (Conv2d)  | 3           | 1      | 1       | 3x3             |
| Layer 2 (Conv2d)  | 3           | 1      | 1       | 5x5             |
| Layer 3 (MaxPool2d)| 2         | 2      | 0       | 6x6             |
| Layer 4 (Conv2d)  | 3           | 1      | 1       | 10x10           |
| Layer 5 (Conv2d)  | 3           | 1      | 1       | 14x14           |
| Layer 6 (MaxPool2d)| 2         | 2      | 0       | 16x16           |
| Layer 7 (Conv2d)  | 3           | 1      | 0       | 24x24           |
| Layer 8 (Conv2d)  | 3           | 1      | 0       | 32x32           |
| Layer 9 (Conv2d)  | 3           | 1      | 0       | 40x40           |




### Model 2

[Provide a brief description of Model 2 architecture and its key features]

### Model 3

[Provide a brief description of Model 2 architecture and its key features]

### Model 4

[Provide a brief description of Model 2 architecture and its key features]

### Model 5

[Provide a brief description of Model 2 architecture and its key features]

### Model 6

[Provide a brief description of Model 2 architecture and its key features]

### Model 7

[Provide a brief description of Model 2 architecture and its key features]

### Model 8

[Provide a brief description of Model 8 architecture and its key features]


## Results

[Provide a summary of the results obtained from the experiments, including the final train and test accuracies for each model]

## Usage

[Provide instructions on how to run the code and reproduce the results]

