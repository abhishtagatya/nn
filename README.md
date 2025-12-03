
# Neural Network From Scratch – Fashion-MNIST & XOR

This project implements a **fully custom neural network framework in C++**, including matrix operations, layers, activations, forward/backward propagation, and optimizers (Adam/SGD).  
Two demos are included:

- **Fashion-MNIST classifier** (dense network, ReLU, softmax cross-entropy)  
- **XOR learner** (simple MLP using sigmoid + MSE)

---

## Features

- Pure C++ neural-network implementation (no external ML libraries)
- Custom:
  - Matrix class
  - Dense layers
  - Activation functions
  - Loss functions (MSE, Softmax CE)
  - Adam optimizer
- Mini-batch training
- CSV-based dataset loading
- Outputs CSV prediction dumps for training/testing sets

---

## Requirements

- C++17 or later  
- Fashion-MNIST data in CSV format:
  - `fashion_mnist_train_vectors.csv`
  - `fashion_mnist_train_labels.csv`
  - `fashion_mnist_test_vectors.csv`
  - `fashion_mnist_test_labels.csv`

Place them in the `data/` directory.

---

## Running the FMNIST Trainer

```bash
./YourExecutable <epochs> <learning_rate> <batch_size>
```

or

```
sh run.sh
```

## Author

- Vladimír Žbánek, 
- Abhishta Gatya Adyatma
