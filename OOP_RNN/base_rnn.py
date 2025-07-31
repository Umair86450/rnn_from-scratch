# base_rnn.py
from abc import ABC, abstractmethod
import random
import math

class RNNBase(ABC):
    def __init__(self, vocab_size, hidden_size):
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size

        # Private weight and bias matrices
        self._Wx = [[random.uniform(-0.1, 0.1) for _ in range(vocab_size)] for _ in range(hidden_size)]
        self._Wh = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self._Wy = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(vocab_size)]

        self._bh = [0] * hidden_size
        self._by = [0] * vocab_size

    # Utility Functions
    @staticmethod
    def one_hot(index, size):
        vec = [0] * size
        vec[index] = 1
        return vec

    @staticmethod
    def matvec_mul(matrix, vector):
        return [sum(m * v for m, v in zip(row, vector)) for row in matrix]

    @staticmethod
    def vec_add(v1, v2):
        return [a + b for a, b in zip(v1, v2)]

    @staticmethod
    def tanh(vec):
        return [math.tanh(v) for v in vec]

    @staticmethod
    def softmax(x):
        max_x = max(x)
        e_x = [math.exp(i - max_x) for i in x]
        total = sum(e_x)
        return [i / total for i in e_x] if total else [1.0 / len(x)] * len(x)

    @staticmethod
    def cross_entropy(predicted, target_idx):
        return -math.log(predicted[target_idx] + 1e-9)

    @abstractmethod
    def train(self, data, word_to_idx):
        pass

    @abstractmethod
    def predict(self, sentence, word_to_idx, idx_to_word):
        pass
