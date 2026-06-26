# Quantum Binary Classifier with PennyLane

A variational quantum classifier (VQC) for binary classification tasks, implemented with [PennyLane](https://pennylane.ai/). The model encodes classical data into quantum states, applies a parametrized quantum circuit, and optimizes the parameters via gradient descent using the parameter-shift rule.

## How it works

### 1. Data encoding — Angle Encoding

Classical features are embedded into quantum states using rotation gates. Each feature $x_i$ is encoded as the angle of an $R_Y$ gate on qubit $i$:

$$R_Y(x_i)|0\rangle = \cos\frac{x_i}{2}|0\rangle + \sin\frac{x_i}{2}|1\rangle$$

Features are scaled to $[0, \pi]$ before encoding so that the full range of the Bloch sphere's Y-axis is used. This means **one qubit per feature**, so the number of qubits is inferred automatically from the dataset.

### 2. Ansatz — Parametrized Quantum Circuit

After encoding, $L$ layers of the following structure are applied:

```
|0⟩ ── RY(x₀) ── RY(θ₀) ──●──
|0⟩ ── RY(x₁) ── RY(θ₁) ──⊕──
```

Each layer consists of:
- $R_Y(\theta_{i,j})$ rotations on each qubit, with trainable parameters $\theta$
- CNOT gates between adjacent qubits to create entanglement

Entanglement is what allows the circuit to learn correlations between features that a product state could not capture.

### 3. Measurement and classification

The output is the expectation value of the Pauli-Z operator on qubit 0:

$$\hat{y} = \langle Z_0 \rangle \in [-1, +1]$$

Classification is done by taking the sign: $\text{sign}(\hat{y}) \in \{-1, +1\}$.

### 4. Training — Parameter-shift rule

The loss function is the mean squared error between predictions and labels:

$$\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2$$

Gradients are computed analytically via the **parameter-shift rule**, which is exact (not a finite-difference approximation):

$$\frac{\partial \langle Z \rangle}{\partial \theta} = \frac{1}{2}\left[\langle Z \rangle_{\theta + \pi/2} - \langle Z \rangle_{\theta - \pi/2}\right]$$

Parameters are updated with the **Adam optimizer**.

## Project structure

```
qml_classifier/
├── data.py        # Dataset generation and preprocessing
├── circuito.py    # Quantum circuit (encoding + ansatz)
├── train.py       # Training loop, accuracy, CLI
└── plot.py        # Decision boundary and loss curve plots
```

## Supported datasets

| Dataset | Features | Qubits | Description |
|---------|----------|--------|-------------|
| `moons` | 2 | 2 | Two interleaved half-moons |
| `xor` | 2 | 2 | XOR pattern in four quadrants |
| `circles` | 2 | 2 | Concentric circles |
| `iris` | 4 | 4 | Versicolor vs Virginica (Iris dataset) |

## Usage

```bash
# Basic run with defaults (moons, 2 layers, 50 epochs, lr=0.1)
python train.py

# Custom run
python train.py --dataset circles --layers 3 --epochs 100 --lr 0.05 --seed 42

# Show dataset plot
python train.py --dataset xor --plot True

# Show circuit diagram
python train.py --circ True
```

### CLI arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | `moons` | Dataset: `moons`, `xor`, `circles`, `iris` |
| `--layers` | int | `2` | Number of ansatz layers |
| `--epochs` | int | `50` | Training epochs |
| `--lr` | float | `0.1` | Adam learning rate |
| `--seed` | int | `123` | Random seed |
| `--plot` | bool | `False` | Plot the dataset |
| `--circ` | bool | `False` | Save circuit diagram |

## Results

On `make_moons` with 2 layers and 50 epochs:

- **Test accuracy: 87.5%**
- The decision boundary learned is non-linear, demonstrating that the VQC can separate classes that are not linearly separable.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install pennylane scikit-learn matplotlib numpy
```

## Dependencies

- [PennyLane](https://pennylane.ai/) — quantum circuit simulation and differentiation
- [scikit-learn](https://scikit-learn.org/) — datasets and preprocessing
- [NumPy](https://numpy.org/) — numerical computing
- [Matplotlib](https://matplotlib.org/) — visualization

## References

- Schuld, M. & Petruccione, F. (2021). *Machine Learning with Quantum Computers*. Springer.
- Mitarai, K. et al. (2018). Quantum circuit learning. *Physical Review A*, 98, 032309.
- PennyLane documentation: https://docs.pennylane.ai
