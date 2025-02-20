# 🧠 Deep Learning

## 📖 Introduction
Deep Learning is a subset of Machine Learning that uses neural networks with multiple layers to automatically learn representations from data. It powers applications such as image recognition, natural language processing, and autonomous systems.

This repository covers:
- **Fundamentals of Deep Learning**
- **Forward & Backward Propagation**
- **Activation Functions**
- **Optimizers**
- **Deep Learning Architectures** (ANN, CNN, RNN)

---

## 🔥 1. Neural Networks Overview

A **Neural Network** consists of:
- **Input Layer** (Takes input features)
- **Hidden Layers** (Performs computations)
- **Output Layer** (Generates predictions)
- **Weights & Biases** (Adjustable parameters)
- **Activation Functions** (Adds non-linearity)

---

## 🚀 2. Forward Propagation

### **Mathematical Formula**
For a **single-layer neural network**:

\[
Z = W X + b
\]

\[
A = \sigma(Z)
\]

Where:
- \( W \) = Weight matrix  
- \( X \) = Input matrix  
- \( b \) = Bias  
- \( Z \) = Linear transformation  
- \( A \) = Activation function output  
- \( \sigma \) = Activation function (e.g., Sigmoid, ReLU)

---

## 🔄 3. Backward Propagation

Backward Propagation helps update weights to minimize the loss function using **Gradient Descent**.

### **Mathematical Formulas**
#### **Loss Function (Mean Squared Error - MSE)**
\[
L = \frac{1}{m} \sum (Y - A)^2
\]

#### **Gradient Descent Update Rules**
\[
W = W - \alpha \frac{\partial L}{\partial W}
\]

\[
b = b - \alpha \frac{\partial L}{\partial b}
\]

Where:
- \( \alpha \) = Learning rate
- \( \frac{\partial L}{\partial W} \), \( \frac{\partial L}{\partial b} \) = Gradients of loss

---

## 🔥 4. Activation Functions

Activation functions introduce non-linearity into neural networks.

### **Types of Activation Functions**

| Activation Function | Formula | Range | Pros | Cons |
|---------------------|---------|-------|------|------|
| **Sigmoid** | \( \sigma(Z) = \frac{1}{1 + e^{-Z}} \) | (0,1) | Good for probability-based outputs | Vanishing gradient problem |
| **ReLU** | \( f(Z) = \max(0, Z) \) | (0,∞) | Solves vanishing gradient | Can cause dead neurons |
| **Tanh** | \( f(Z) = \frac{e^Z - e^{-Z}}{e^Z + e^{-Z}} \) | (-1,1) | Zero-centered | Still suffers from vanishing gradients |
| **Leaky ReLU** | \( f(Z) = \max(0.01Z, Z) \) | (-∞,∞) | Fixes dead neurons issue | May not always improve performance |
| **Softmax** | \( \sigma(Z)_i = \frac{e^{Z_i}}{\sum_{j} e^{Z_j}} \) | (0,1) | Useful for multi-class classification | Computationally expensive |

---

## 🚀 5. Optimizers in Deep Learning

Optimizers adjust weights to minimize loss. 

### **Types of Optimizers**
1. **Gradient Descent (GD)**
   - Updates weights using **entire dataset**
   - Converges slower

2. **Stochastic Gradient Descent (SGD)**
   - Updates weights using **one sample at a time**
   - Noisy updates but faster convergence

3. **Mini-Batch Gradient Descent**
   - Updates weights using **a small batch** of samples
   - Best of both worlds

4. **Adam (Adaptive Moment Estimation)**
   - Combines momentum and adaptive learning rate
   - **Formula:**
     \[
     m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
     \]
     \[
     v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
     \]
   - Faster convergence

5. **RMSprop (Root Mean Square Propagation)**
   - Reduces learning rate for frequently occurring features
   - Prevents oscillations

---

## 🏛️ 6. Deep Learning Architectures

### 6.1 **Artificial Neural Network (ANN)**
- **Fully connected network** with input, hidden, and output layers.
- Used for structured/tabular data problems.
- Forward & Backward Propagation applies.

![ANN](https://upload.wikimedia.org/wikipedia/commons/e/e4/Artificial_neural_network.svg)

---

### 6.2 **Convolutional Neural Network (CNN)**
- Used for **image recognition** tasks.
- Contains **Convolutional layers**, **Pooling layers**, and **Fully connected layers**.

#### **Key Components**
1. **Convolutional Layer** - Extracts features using filters/kernels.
2. **ReLU Activation** - Introduces non-linearity.
3. **Pooling Layer (Max/Average Pooling)** - Reduces dimensionality.
4. **Fully Connected Layer** - Makes final classification.

#### **Mathematical Formula for Convolution**
\[
Z = W * X + b
\]
Where \( * \) is the convolution operation.

---

### 6.3 **Recurrent Neural Network (RNN)**
- Used for **sequential data** (time series, speech, text).
- Stores **hidden states** to process previous inputs.

#### **Formula for RNN Cell**
\[
h_t = \tanh(W_h h_{t-1} + W_x X_t + b)
\]

- Problem: **Vanishing Gradient** in long sequences.
- Solution: **LSTM & GRU** (Long Short-Term Memory & Gated Recurrent Unit).

#### **LSTM Formula**
\[
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
\]

\[
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
\]

\[
C_t = f_t \cdot C_{t-1} + i_t \cdot \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
\]

---

## 🛠️ 7. Installation & Setup

To install dependencies:

```bash
pip install numpy matplotlib tensorflow keras
