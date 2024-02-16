# XOR Neural Network in C

## Overview
This project implements a basic XOR Neural Network using C. The network consists of an input layer, one hidden layer, and an output layer. It demonstrates the fundamental concepts of neural networks, including forward propagation, backpropagation, the use of activation functions, and training with momentum.

## Features
- Simple XOR logic implementation.
- Use of Sigmoid and Hyperbolic Tangent (Tanh) as activation functions.
- Backpropagation algorithm to adjust weights and biases.
- Momentum to accelerate convergence.

## Requirements
- GCC Compiler (or any C compiler)
- Basic understanding of neural networks and C programming

## Compilation and Execution
Compile the program using the following command:

gcc -o xor_nn xor_nn.c -lm

./xor_nn


## How It Works
The program initializes a neural network with randomly assigned weights and biases. It then trains the network using a dataset that represents the XOR logic table. The training process involves forward propagation to predict outputs, calculation of errors, and backpropagation to adjust weights and biases. The program outputs the mean squared error at the end of the training and tests the trained model with the XOR dataset.

## Contributing
Feel free to fork this project, make changes, and submit pull requests. Suggestions for improving the neural network's efficiency and functionality are welcome.

## License
This project is open-source and licensed under the Apache License 2.0. For more details, see the LICENSE file in this repository.

