# Fundamental Machine Learning Algorithms from Scratch in Python

A repository of core Machine Learning algorithms, optimization techniques, and mathematical formulations implemented from scratch using Python and [NumPy](https://numpy.org/). The codebase covers regression models, gradient descent optimizers, single and multi-layer neural networks with backpropagation, linear classifiers, support vector geometry, and unsupervised clustering with real-world robotics problem scenarios.

---

## Implemented Algorithms & Modules

| Script | Category | Method / Technique | Problem Scenario |
|---|---|---|---|
| [`linear_regression.py`](linear_regression.py) | Regression | Ordinary Least Squares (OLS) closed-form solution | Analytical regression fitting |
| [`linear_regression_numpy.py`](linear_regression_numpy.py) | Regression | Vectorized Ordinary Least Squares via NumPy matrix algebra | Multi-variable linear regression |
| [`gradient_decent_linear_regression.py`](gradient_decent_linear_regression.py) | Optimization | Batch Gradient Descent with MSE cost tracking | Iterative parameter optimization |
| [`polynomial_regression_numpy.py`](polynomial_regression_numpy.py) | Regression | Polynomial feature mapping & Vandermonde matrix fitting | Nonlinear curve fitting |
| [`perceptron.py`](perceptron.py) | Classification | Rosenblatt Perceptron learning rule with Heaviside step activation | Linearly separable binary classification |
| [`gadient_descent_sigmoid_perceptron.py`](gadient_descent_sigmoid_perceptron.py) | Classification | Logistic Perceptron with Sigmoid activation & Cross-Entropy/MSE loss | Probabilistic binary classification |
| [`backprobagation.py`](backprobagation.py) | Neural Networks | Multi-layer Perceptron (MLP) forward and backward propagation | Error backpropagation & weight updates |
| [`backprobagation_Mix_A_F.PY`](backprobagation_Mix_A_F.PY) | Neural Networks | Backpropagation network with mixed activation functions | Layer-specific non-linear transformations |
| [`SVM.py`](SVM.py) | Classification | Support Vector Machine hyperplane geometry & margin optimization | UR5e robotic gripper grasp classification |
| [`clustering.py`](clustering.py) | Unsupervised | Iterative K-Means clustering algorithm with convergence checking | Pioneer 3-DX LIDAR obstacle mapping |

---

## Detailed Methodologies

### 1. Neural Networks & Backpropagation
- **Forward Path**: Propagates input activations through affine linear combinations ($z = Wx + b$) and non-linear activation functions ($\sigma(z) = \frac{1}{1 + e^{-z}}$).
- **Backward Path**: Computes partial derivatives of the loss function $E = \frac{1}{2}\sum (d_k - y_k)^2$ using the chain rule:
  $$\delta_j = \frac{\partial E}{\partial a_j} = \frac{\partial E}{\partial y_j} \cdot \sigma'(a_j)$$
  $$\Delta w_{ij} = \eta \cdot \delta_j \cdot x_i$$

### 2. Support Vector Machine Geometry (`SVM.py`)
- Programmatically calculates the geometric margin and perpendicular distance ($d$) from support vectors to the decision boundary:
  $$d = \frac{|\mathbf{w}^T \mathbf{x} - b|}{\|\mathbf{w}\|}$$
- Evaluates the primal objective function minimizing $\frac{1}{2} \|\mathbf{w}\|^2$ to maximize the separation margin for robotic grip state verification.

### 3. Obstacle Clustering (`clustering.py`)
- Implements the complete expectation-maximization cycle of **K-Means Clustering**:
  1. **Assignment Phase**: Measures Euclidean distance $\|x_i - \mu_k\|$ from LIDAR obstacles to cluster centroids.
  2. **Update Phase**: Recomputes centroid positions as the arithmetic mean of assigned points.
  3. **Convergence Check**: Terminates iterations when centroid positions stabilize ($\Delta \mu = 0$).

### 4. Gradient Descent Optimization
- Computes empirical loss gradients across parameter space:
  $$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$$
- Features adaptive learning rate evaluation and iterative convergence visualization.

---

## Project Structure

```text
Machine-Learning-Codes-in-Python/
├── backprobagation.py                   # 2-layer neural network with Sigmoid backpropagation
├── backprobagation_Mix_A_F.PY           # Neural network with mixed layer activation functions
├── clustering.py                        # From-scratch K-Means clustering for mobile robot LIDAR
├── gadient_descent_sigmoid_perceptron.py# Gradient descent training on Sigmoid perceptron
├── gradient_decent_linear_regression.py # Iterative gradient descent for linear regression
├── linear_regression.py                 # Analytic Ordinary Least Squares implementation
├── linear_regression_numpy.py           # Vectorized linear regression using NumPy
├── perceptron.py                        # Classic Rosenblatt perceptron learning algorithm
├── polynomial_regression_numpy.py       # Polynomial curve fitting with feature expansion
├── SVM.py                               # SVM hyperplane margin and distance formulation
└── README.md
```

---

## Prerequisites & Execution

### Requirements
- Python 3.8 or newer
- NumPy:
  ```bash
  pip install numpy
  ```

### Running Individual Algorithms

Each module is self-contained and can be executed directly from the terminal:

```bash
# Run K-Means LIDAR obstacle clustering
python clustering.py

# Run Neural Network Backpropagation demonstration
python backprobagation.py

# Run Support Vector Machine margin calculations
python SVM.py

# Run Linear Regression with Gradient Descent
python gradient_decent_linear_regression.py
```

---

## Author

- **Mohamed Ghanem** - [Eng-Ghanem](https://github.com/Eng-Ghanem)
