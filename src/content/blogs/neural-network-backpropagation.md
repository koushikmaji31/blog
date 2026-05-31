---
title: "The Mathematics of Backpropagation"
excerpt: "Exploring the chain rule and how gradients flow backward through neural networks."
date: "2024-03-10"
readTime: "12 min read"
category: "Deep Learning"
---

## Contents

# The Mathematics of Backpropagation

Backpropagation is the algorithm that makes training deep neural networks possible. Let's understand the mathematics behind it.

## The Chain Rule Foundation

Backpropagation is fundamentally an application of the chain rule from calculus. For a composite function $f(g(x))$, the chain rule states:

$$\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

## Neural Network Setup

Consider a simple neural network with layers:
- Input layer: $x$
- Hidden layer: $h = \sigma(W_1 x + b_1)$
- Output layer: $\hat{y} = W_2 h + b_2$
- Loss function: $L = ||y - \hat{y}||^2$

Where $\sigma$ is an activation function (e.g., sigmoid, ReLU).

## Forward Pass

The forward pass computes:

$$z_1 = W_1 x + b_1$$
$$h = \sigma(z_1)$$
$$z_2 = W_2 h + b_2$$
$$\hat{y} = z_2$$
$$L = \frac{1}{2}||y - \hat{y}||^2$$

## Backward Pass

Now we compute gradients working backwards:

### Output Layer Gradients

$$\frac{\partial L}{\partial \hat{y}} = \hat{y} - y$$

$$\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial \hat{y}} \cdot h^T$$

$$\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial \hat{y}}$$

### Hidden Layer Gradients

$$\frac{\partial L}{\partial h} = W_2^T \cdot \frac{\partial L}{\partial \hat{y}}$$

$$\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial h} \odot \sigma'(z_1)$$

Where $\odot$ denotes element-wise multiplication.

### Input Layer Gradients

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z_1} \cdot x^T$$

$$\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial z_1}$$

## Computational Efficiency

The key insight of backpropagation is that we can reuse computations:

1. Each gradient builds on previously computed gradients
2. We store intermediate values from the forward pass
3. Time complexity is $O(W)$ where $W$ is the number of weights

## Common Activation Functions and Their Derivatives

### Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

### ReLU

$$\text{ReLU}(x) = \max(0, x)$$

$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}$$

### Tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

$$\tanh'(x) = 1 - \tanh^2(x)$$

## Conclusion

Backpropagation efficiently computes gradients by applying the chain rule systematically through the network. This elegant algorithm is what makes training deep networks tractable.
