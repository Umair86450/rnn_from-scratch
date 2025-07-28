
# 🧠 RNN from Scratch in Pure Python

This project implements a **Recurrent Neural Network (RNN)** from scratch using **only core Python**, with **manual forward and backward propagation** (no libraries like PyTorch or TensorFlow). It is designed to help beginners understand the internals of how RNNs work — especially the math and logic behind training using **backpropagation through time**.

---

## 📚 Dataset

A few sample English sentences used for next-word prediction:

```

"hello how are you",
"how are you doing",
"are you doing well"

```

The goal: Given a word like `"hello"`, predict the next word (`"how"`), and so on.

---

## 🔧 Project Structure

```

rnn-from-scratch/
├── rnn.py             # Full Python implementation (forward + backprop)
├── README.md          # This file

````

---

## 🚀 How to Run

1. Clone this repository:

```bash
git clone https://github.com/your-username/rnn-from-scratch.git
cd rnn-from-scratch
````

2. Run the code:

```bash
python3 rnn.py
```

---

## 🧠 How It Works

### 🟦 Forward Pass

At each time step:

1. Compute hidden state:

   $$
   h_t = \tanh(W_x \cdot x + W_h \cdot h_{t-1} + b_h)
   $$

2. Compute output:

   $$
   y = W_y \cdot h_t + b_y
   $$

3. Apply softmax to get probabilities:

   $$
   \hat{y} = \text{softmax}(y)
   $$

---

### 🔁 Backpropagation Through Time (BPTT)

The backward pass uses **chain rule** to propagate gradients back through the network:

#### Loss Function:

$$
\text{Cross-Entropy Loss: } L = -\log(\hat{y}_{\text{true}})
$$

#### Gradients:

* **Output layer:**

  $$
  \frac{\partial L}{\partial y_i} = \hat{y}_i - y_{\text{true},i}
  $$

* **Hidden state:**

  $$
  \frac{\partial L}{\partial h_t} = W_y^T \cdot (\hat{y} - y)
  $$

* **Through tanh:**

  $$
  \frac{d}{dz} \tanh(z) = 1 - \tanh^2(z)
  $$

* **Input & hidden weights:**

  $$
  \frac{\partial L}{\partial W_x} = \delta \cdot x^T, \quad
  \frac{\partial L}{\partial W_h} = \delta \cdot h_{t-1}^T
  $$

---

## 📈 Output Example

After training, the RNN predicts the next word in each sentence:

```
Input: 'hello' → Predicted: 'how' (Target: 'how')
Input: 'how' → Predicted: 'are' (Target: 'are')
Input: 'are' → Predicted: 'you' (Target: 'you')
...
```

---

## 🛠 Features

* Pure Python — no NumPy, PyTorch, or TensorFlow
* Manual forward pass and backpropagation
* Word prediction with one-hot encoding
* Tanh activation and softmax output
* Cross-entropy loss
* Fully commented code for learning

---




## ✅ Todo

* [ ] Add batching support
* [ ] Add support for LSTM / GRU
* [ ] Load external text dataset (e.g. WikiText)
* [ ] Save/load model weights
