# Relation Extraction from Natural Language using PyTorch

This project aims to extract relationships from natural language using PyTorch. The dataset used is a collection of text snippets and their associated labels, which represent the relationships in the text.

## Requirements
* sklearn
* pandas
* torch
* seaborn
* matplotlib
* numpy


## Data Preprocessing

The dataset is loaded and the labels are cleaned by replacing "none" with an empty string and replacing any missing values with an empty string. The labels are then split into a list of strings. The text and labels are split into training and testing sets using `train_test_split` from scikit-learn. The text is then transformed into a tf-friendly format using the `TfidfVectorizer`. The labels are transformed using `MultiLabelBinarizer` from scikit-learn.

## Dataset Preparation

A PyTorch `Dataset` class is created to hold the training and testing data. The `DataLoader` class is used to load the data in batches for training and testing.

## Model and Training

A multi-layered perceptron (MLP) classifier is implemented using PyTorch. The model consists of two fully connected layers with ReLU activation. The model is trained using the Adam optimizer and cross entropy loss. The model's performance is evaluated on the testing set after each epoch.

## Testing

The model is tested on a separate test set and the performance is reported in terms of precision, recall, and F1 score.

## Usage

To use the code, specify the file paths for the training and testing datasets in the appropriate lines. The model can then be trained and tested by running the code. The model's hyperparameters and other settings can be modified as desired.
