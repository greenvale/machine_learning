# Machine Learning projects
This repository contains my machine learning projects, such as implementations of machine learning models.

## Computational graph (C++)
For pedagogical purposes, I have implemented a computational graph class with automatic differentation. The computational graph operates on vectors encapsulated by `std::vector` and all operations are vectorised. Key operations include matrix-vector multiplication, vector summation, ReLU and softmax.

For the forward pass:
* Output vectors are stored within each operation class.
* Input vectors for a node are stored as pointers to vectors that either belong as outputs in a parent node or are standalone vectors.
* Data flows forward from parent nodes to child nodes

For backpropagation:
* Gradients are outputted from child nodes to parent nodes
* Each node sums the gradient inputs from child nodes and then applies its functional derivative

The computational graph class was used to build a neural network with 1 hidden layer. This was then trained on the MNIST dataset using stochastic gradient descent -- achieving an accuracy of ~91%.

## Reinforcement Learning
### Fruit Catch
'Fruit Catch' is a simple arcade-style game that I have created using Pygame. The game can either be played manually using keyboard input or can be controlled by an agent.

I have implemented a Q-reinforcement learning agent using a lookup table Q function. This is only configured to play the game when there is a single fruit falling at any given time. The average success rate was recorded to be ~98%.

I am implementing a deep Q network using a convolutional neural network that is capable of playing the game when there are multiple fruit.

## CNN
I have implemented a convolutional neural network using PyTorch that is trained on the MNIST dataset, achieving an accuracy of ~97%.
