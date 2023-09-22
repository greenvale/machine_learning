import torchvision
from torchvision import transforms
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
import numpy as np
import math

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
testset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Utility function -- plots a 3d tensor as a grid of 2d imgs
def imshow_3d(tensor_3d : torch.tensor, axis3 : int = 0):
    assert(2 <= tensor_3d.ndim <= 3)
    if tensor_3d.ndim == 2:
        plt.imshow(tensor_3d)
    else:
        assert(axis3 >= 0)
        T = tensor_3d
        if axis3 > 0:
            T = tensor_3d.moveaxis(axis3, 0)
    
        # get num rows and cols
        num_rows = int(math.sqrt(T.shape[0]))
        while T.shape[0] % num_rows:
            num_rows -= 1
        num_cols = int(T.shape[0] / num_rows)
    
        # plot
        fig, axs = plt.subplots(num_rows, num_cols)
        for i in range(num_rows):
            for j in range(num_cols):
                axs[i, j].imshow(T[i*num_cols+j].detach().numpy())
                axs[i, j].set(title = f"Img {i*num_cols+j}")
        plt.show(fig)

# CNN model
class Model(nn.Module):
    def __init__(self, seed=42):
        torch.manual_seed(seed)
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(200, 10)

    def forward(self, x, y=None):
        self.a1 = self.conv1(x)
        self.b1 = self.pool1(self.a1)
        self.a2 = self.conv2(self.b1)
        self.b2 = self.pool2(self.a2)
        self.f = self.flat(self.b2)
        self.logits = self.lin1(self.f)
        pred = F.softmax(self.logits, dim=-1)
        if y is not None:
            loss = F.cross_entropy(self.logits, y)
            return pred, loss
        else:
            return pred

# train the model using minibatch SGD
# plot the smoothed loss function if requested
def train_model(model, batch_size:int = 32, learning_rate:float = 0.001, num_batches:int = 10000, val_freq:int=200, plot_loss:bool = False):
    val_ts = []
    val_history = []
    loss_ts = []
    loss_history = []

    # get train/val sets
    split = 0.9
    split_n = int(trainset.data.shape[0] * split)
    X_tr = trainset.data[:split_n].unsqueeze(1) / 255.0
    y_tr = trainset.targets[:split_n]
    X_val= trainset.data[split_n:].unsqueeze(1) / 255.0
    y_val= trainset.targets[split_n:]

    # set model mode to training and train
    model.train()
    for n in range(num_batches):
        # get batch
        idx = torch.randint(0, split_n, (batch_size,))
        # X img: normalise and reshape (28,28)-->(1,28,28)
        X = X_tr[idx]
        y = y_tr[idx]

        #print(f"Xshape={list(X.shape)}, yshape={list(y.shape)}")

        # forward pass
        pred, loss = model.forward(X, y)

        # record loss with smoothing
        if len(loss_history) > 0:
            loss_history.append(0.01*loss.item() + 0.99*loss_history[-1])
        else:
            loss_history.append(loss.item())
        loss_ts.append(n)

        # calculate and record val loss
        if n % val_freq == 0:
            pred, loss = model.forward(X_val, y_val)
            if len(val_history) > 0:
                val_history.append(loss.item())
            else:
                val_history.append(loss.item())
            val_ts.append(n)

        # backprop
        for p in model.parameters():
            p.grad = None
        loss.backward()
        for p in model.parameters():
            p.data += -learning_rate * p.grad

    if plot_loss is True:
        fig = plt.figure()
        ax = fig.add_axes((0,0,1,1))
        ax.plot(loss_ts, loss_history)
        ax.plot(val_ts, val_history)
        ax.set_xlabel("Batch")
        ax.set_ylabel("Loss")
        ax.legend(labels=["Train","Val"])
        plt.show()

# test the model by passing through the entire testset
# calculate the accuracy
def test_model(model):
    model.eval()
    X_test = testset.data.unsqueeze(1) / 255.0
    pred = model.forward(X_test)
    acc = (pred.argmax(dim=1) == testset.targets).sum().item() / float(testset.targets.shape[0])
    return acc

# main
if __name__ == "__main__":
    batch_size = 32
    lr = 0.01
    num_batches = 10_000
    val_freq = 400

    model = Model()
    
    train_model(model, batch_size, lr, num_batches, val_freq, True)
    
    acc = test_model(model)
    print(f"accuracy={acc}")