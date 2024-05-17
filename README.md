# DL Assignment 3
# Seq2Seq Transliteration System
This repository contains the implementation of a Seq2Seq model for a transliteration system, with two variants: one without attention and one with attention mechanism. The models are implemented using PyTorch.

## Introduction
The Seq2Seq transliteration system is designed to convert text from one script to another. The project includes two implementations:

- Without Attention: A simple Seq2Seq model.
- With Attention: A Seq2Seq model enhanced with an attention mechanism.

## Requirements
To run the code, you need the following libraries:
- Python 3.7+
- torch
- pandas
- wandb
- argparse
  
## Libraries Used
- Wandb: Used for experiment tracking and visualization.
- PyTorch: Deep learning framework for building and training the CNN model.
- numpy: Library for numerical computing.
## Dataset
Prepare the dataset and create three files containing the data -- train.csv, test.csv, validation.csv.

## Training the Model
### Command Line Arguments
```
-wp, --wandb_project            :Specifies the project name for Weights & Biases.
-e, --epochs                    :Number of epochs to train the model.
-lr, --learning_rate            :Learning rate for the optimizer.
-b, --batch_size                :Batch size for training.
-embd_dim, --char_embd_dim      : Dimension of character embeddings.
-hid_neur, --hidden_layer_neurons: Number of neurons in hidden layers.
-num_layers, --number_of_layers : Number of layers in the encoder and decoder.
-cell, --cell_type              : Type of RNN cell (RNN, LSTM, GRU).
-do, --dropout                  : Dropout probability.
-opt, --optimizer               : Optimization algorithm (adam, nadam).
-train_path, --train_path       :(required) Path to the training data CSV file.
-test_path, --test_path         :(required) Path to the testing data CSV file.
-val_path, --val_path           : (required) Path to the validation data CSV file.
```

## Model Architecture
### Without Attention
The train_without_attention.py script implements a basic Seq2Seq model consisting of an encoder and a decoder. The encoder encodes the input sequence into a context vector, which is then used by the decoder to generate the output sequence.

### With attention
The train_with_attention.py script implements a Seq2Seq model with an attention mechanism. This model allows the decoder to focus on different parts of the input sequence at each step, improving the performance for longer sequences.
##  Link to Report

The wandb report is available [here](https://wandb.ai/cs23m021/DL_Assignment_2/reports/CS6910-Assignment-2--Vmlldzo3NTA4NjE5).

##  Acknowledgements

For the most of assignment, I have referred the lecture slides of Dr. Mitesh Khapra, Indian Institute of Technology Madras: [CS6910 Lecture Slides](http://cse.iitm.ac.in/~miteshk/CS6910.html#schedule)

Special thanks to:
- [GitHub](https://github.com/)
- [Weights & Biases (wandb)](https://wandb.ai)

These resources were invaluable in the development and understanding of the assignment.
