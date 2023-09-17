import math
import pandas as pd
import torch
import numpy as np

# Helper functions
def get_class_counts(y, num_classes : int) -> list[int]:
    return [ (y[y==i].shape)[0] for i in range(num_classes) ]

def get_probs(y, num_classes : int, offset : float) -> torch.tensor:
    probs = torch.tensor(get_class_counts(y, num_classes), dtype=torch.float)
    probs += offset
    probs /= probs.sum().item()
    return probs

def get_entropy(y, num_classes : int, offset : float) -> float:
    probs = get_probs(y, num_classes, offset)
    return float(-(probs.log2() * probs).sum().item())

# Leaf node
class LeafNode:
    def __init__(self, num_classes : int):
        self.num_classes = num_classes
        self.probs_offset = 0.001
        self.probs = None
        self.y = None
    
    def eval(self, X : pd.DataFrame):
        if X.shape[0] > 0:
            self.y = torch.multinomial(self.probs, X.shape[0], replacement=True)
        else:
            self.y = torch.tensor([])
    
    def train(self, X : pd.DataFrame, y : pd.DataFrame):
        self.probs = get_probs(y, self.num_classes, self.probs_offset)

# Decision node
class DecisionNode:
    def __init__(self, num_classes : int):
        self.num_classes = num_classes
        self.feature = None
        self.threshold = None
        self.lhs = None
        self.rhs = None
        self.probs_offset = 0.001
        self.info_gain = None

    def eval(self, X : pd.DataFrame):
        assert(self.feature in X.columns)
        if self.lhs:
            self.lhs.eval(X[X[self.feature] <= self.threshold])
        if self.rhs:
            self.rhs.eval(X[X[self.feature] > self.threshold])

    def train(self, X : pd.DataFrame, y : pd.DataFrame):
        start_entropy = get_entropy(y, self.num_classes, self.probs_offset)

        best_feature = None
        best_threshold = None
        best_entropy = None

        ranges = { col:[X[col].min(), X[col].max()] for col in X.columns }

        for col in X.columns:
            for x in list(np.linspace(ranges[col][0], ranges[col][1], 10)):
                y_left  = y[X[col] <= x.item()]
                y_right = y[X[col] >  x.item()]

                entropy_lhs = get_entropy(y_left, self.num_classes, self.probs_offset)
                entropy_rhs = get_entropy(y_right, self.num_classes, self.probs_offset)
                entropy = 0.5 * (entropy_lhs + entropy_rhs)

                if (best_feature and (entropy < best_entropy)) or (not best_feature):
                    best_entropy = entropy
                    best_threshold = x
                    best_feature = col

        self.feature = best_feature
        self.threshold = best_threshold.item()
        self.info_gain = -best_entropy + start_entropy

        if self.lhs:
            self.lhs.train(X[X[self.feature] <= self.threshold], y[X[self.feature] <= self.threshold])
        if self.rhs:
            self.rhs.train(X[X[self.feature] > self.threshold], y[X[self.feature] > self.threshold])