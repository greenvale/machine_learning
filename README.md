# Machine Learning projects
This repository contains my machine learning projects, such as implementations of machine learning models.

## C++ implementations  

### Neural Network
Neural networks are used for modelling classification or regression problems. The `neural_network` class is an implementation of a feed-forward neural network in C++ using the standard library vector container. Here is a summary of the theory behind the neural network:

The network has $1$ input layer, $N-1$ hidden layers and $1$ output layer. That means $N$ layers require computing to generate the final output. The shape of network is given by vector $\textbf{l}$ with length $N+1$, where $l^{(i)}$ is the number of nodes in the $i^\text{th}$ layer. The input vector is $\textbf{x}$, with length $l^{(0)}$. The hidden layers are $\textbf{h}^{(1)} \text{, ...  , } \textbf{h}^{(N-1)}$ with respective lengths $l^{(1)}\text{, ... , }l^{(N-1)}$. The output layer is $\textbf{h}^{(N)}$ with length $l^{(N)}$. 

Each layer $\textbf{h}^{(i)}$ is calculated using an activation function. The pre-activated vector for each layer is $\textbf{z}^{(i)}$, where $\textbf{h}^{(i)} = \sigma(\textbf{z}^{(i)})$ for some activation function $\sigma$. The pre-activated vector is a linear function of the previous layer  $\textbf{x}$ or $\textbf{h}^{(1)}\text{, ... ,}\textbf{h}^{(N-1)}$:

$$\textbf{z}^{(i)} = W^{(i)}\textbf{x} + \textbf{b}^{(i)}$$
For $i=1$

$$\textbf{z}^{(i)} = W^{(i)}\textbf{h}^{(i-1)} + \textbf{b}^{(i)}$$
For $1 < i \le N$

Where $W^{(i)}$ is the weight matrix with dimensions $(l^{(i)} \times l^{(i-1)})$ and $\textbf{b}^{(i)}$ is the bias vector with length $l^{(i)}$, for the $i^{\text{th}}$ layer. Note: in the code, the indexing of the layers, pre-activated layers, weights and biases begins at $0$ instead of $1$, as indicated in the theory.

The following activation functions are typically used in neural networks:
* ReLU (Rectified linear unit):
$$\text{ReLU}(x)=\text{max}(x,0)$$
$$\frac{\partial}{\partial x} \text{ReLU}(x)=\text{sgn}(\text{max}(x,0))$$
* Sigmoid
$$\text{sig}(x) = {1 \over 1 + e^{-x}}$$
$$\frac{\partial}{\partial x} \text{sig}(x) = \text{sig}(x) (1 - \text{sig}(x))$$
* Softmax
$$\text{softmax}(x_i,\textbf{x}) = {e^{x_i} \over \sum_{j=1}^{m}e^{x_j}}$$
$$\frac{\partial}{\partial x} \text{softmax}(x_i) = \text{softmax}(x_i) (1 - \text{softmax}(x_i))$$

For $i \in \{1, ... ,m\}$

Note that Softmax is defined over a vector of values and is only used for the output layer with the crossentropy loss function for classification.

The loss function $J$ characterises the accuracy of the neural network by measuring the error of the output against the true values. The neural network is trained by minimising the loss function using gradient descent. For the gradient descent algorithm to achieve a global optimum, the loss function must be convex, with a single stationary point at the point of minimal loss. The loss function used depends on the problem that is being modelled by the neural network: classification or regression:
* Cross-entropy loss (for classification)
$$J(h_j^{(N)},y_j)=y_j\text{log}(h_j^{(N)})+(1-y_j)\text{log}(1-h_j^{(N)})$$
$$\frac{\partial}{\partial h_j^{(N)}} J(h_j^{(N)},y_j)= {h_j^{(N)} - y_j \over h_j^{(N)} (1 - h_j^{(N)})}$$
* Mean-squared error (for regression)
$$J(h_j^{(N)},y_j)={1\over2}(h_j^{(N)} - y_j)^2$$
$$\frac{\partial}{\partial h_j^{(N)}} J(h_j^{(N)},y_j)= h_j^{(N)} - y_j$$
For $j \in \{1, ... , l^{(N)}\}$ for the output layer. 

Ultimately this means the loss function is a function of the model parameters, i.e. the weights and biases:
$$J=J(\textbf{h}^{(N)},\textbf{y})=J(\textbf{x},W^{(1)}, W^{(2)}, ... ,W^{(N)},\textbf{b}^{(1)},\textbf{b}^{(2)}, ... ,\textbf{b}^{(N)},\textbf{y})$$
For a convex positive function $f(x)$, the gradient descent algorithm is used to solve for the $x_0$ s.t. $f'(x_0) = 0$, i.e. the point of local minimum. Consider the first-order Taylor expansion of $f$ about a point $x$ with perturbation $\delta x$:
$$f(x+\delta x) = f(x) + f'(x)\delta x$$
Now suppose we set perturbation $\delta x$ to be $-\alpha f'(x)$ for some small $\alpha \in \mathbb{R}$ then:
$$f(x-\alpha f'(x)) = f(x) - \alpha (f'(x))^2$$
$$\implies 0 < f(x - \alpha f'(x)) < f(x), \forall x \in \mathbb{R}$$
Therefore the gradient descent algorithm updates $x$ such that $f(x)$ converges on the local and thus global minimum for an appropriate $f$. 

In the case of the loss function, the $jk^{\text{th}}$ element of each weight matrices are adjusted by the derivative of the loss function with respect to themselves:

$$\delta W^{(i)}_{jk}=-\alpha \frac{\partial J}{\partial W^{(i)}_{jk}}$$

And similarly for the biases:

$$\delta b^{(i)}_{j}=-\alpha \frac{\partial J}{\partial b^{(i)}_{j}}$$

Where $\alpha$ is the learning rate. 

The calculation of these partial derivatives requires backpropagation, which is underpinned by the chain rule. Each layer is dependent on the previous layers in sequence - this sequence is reversed when using the chain rule. The derivatives with respect to each layer are calculated recurrently:
$$\frac{\partial J}{\partial \textbf{h}^{(i)}} =\frac{\partial J}{\partial \textbf{h}^{(i+1)}} \frac{\partial \textbf{h}^{(i+1)}}{\partial \textbf{z}^{(i+1)}} \frac{\partial \textbf{z}^{(i+1)}}{\partial \textbf{h}^{(i)}}$$

$\frac{\partial \textbf{h}^{(i)}}{\partial \textbf{z}^{(i)}}$ is the analytical derivative of the activation function $\sigma '(\textbf{z}^{(i)})$ and $\frac{\partial \textbf{z}^{(i+1)}}{\partial \textbf{h}^{(i)}} = W^{(i+1)}$. So:

$$\frac{\partial J}{\partial \textbf{h}^{(i)}} =\frac{\partial J}{\partial \textbf{h}^{(i+1)}} \sigma '(\textbf{z}^{(i+1)}) W^{(i+1)}$$

$$\frac{\partial J}{\partial h^{(i)}_j} =\sum_{k=1}^{l^{(i+1)}} \frac{\partial J}{\partial h^{(i+1)}_k} \sigma '(z^{(i+1)}_k) W^{(i+1)}_{kj}$$

For $1 < j < l^{(i)}$

Then the derivatives of the loss with respect to weights and biases is calculated for each layer:

$$\frac{\partial J}{\partial W^{(i)}} =\frac{\partial J}{\partial \textbf{h}^{(i)}} \frac{\partial \textbf{h}^{(i)}}{\partial \textbf{z}^{(i)}} \frac{\partial \textbf{z}^{(i)}}{\partial W^{(i)}}$$

$$\frac{\partial J}{\partial W^{(i)}_{jk}} =\frac{\partial J}{\partial h^{(i)}_j} \frac{\partial h^{(i)}_j}{\partial z^{(i)}_j} \frac{\partial z^{(i)}_j}{\partial W^{(i)}_{jk}} = \frac{\partial J}{\partial h^{(i)}_j}  \sigma '(z^{(i)}_j) x_k$$

For $i=1$

$$\frac{\partial J}{\partial W^{(i)}_{jk}} =\frac{\partial J}{\partial h^{(i)}_j} \frac{\partial h^{(i)}_j}{\partial z^{(i)}_j} \frac{\partial z^{(i)}_j}{\partial W^{(i)}_{jk}} = \frac{\partial J}{\partial h^{(i)}_j}  \sigma '(z^{(i)}_j) h^{(i-1)}_k$$

For $1 < i \le N$

$$\frac{\partial J}{\partial b^{(i)}_{j}}  = \frac{\partial J}{\partial h^{(i)}_j} \frac{\partial h^{(i)}_j}{\partial z^{(i)}_j} \frac{\partial z^{(i)}_j}{\partial b^{(i)}_{j}} = \frac{\partial J}{\partial h^{(i)}_j}  \sigma '(z^{(i)}_j)$$

For $W^{(i)}$, $1 < j < l^{(i)}$ and $1 < k < l^{(i-1)}$ and for $b^{(i)}$, $1 < j < l^{(i)}$.

The $\frac{\partial J}{\partial \textbf{h}^{(i)}}$ derivatives are calculated recurrently. This is initialised at the output layer, where the derivatives are instead calculated using the analytical derivative for the loss function itself.

The training process involves forward-passing the training input through the network and calculating the output, followed by backpropagating the derivatives through the network and adjusting the weights and biases accordingly. The dataset is processed in mini-batches, whereby the adjustments to weights and biases are accumulated during the mini-batch by mean averaging and then applied at the end of each mini-batch.

#### MNIST Dataset
The training example provided in this repo is the classification of handwritten digits 0, ... , 9 in the MNIST dataset. Each input is a 28x28 pixel image that is flattened and normalised. The output is probability distribution across the integers representing the probability that the input image is a given integer. This is compared against a one-hot encoded vector of length 10, representing the correct integer. 

## Python work

In progress.