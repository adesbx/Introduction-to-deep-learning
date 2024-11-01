# Introduction to deep learning

This project was created as a hands-on introduction to **PyTorch** and **hyperparameter tuning** for deep learning models. Three different neural network architectures were built and experimented with to understand their performance and behaviors across various hyperparameter settings.

## Project members
- [CUZIN-RAMBAUD Valentin](https://github.com/valentincuzin)
- [DESBIAUX Arthur](https://github.com/adesbx)

## Architectures

The project includes implementations of the following architectures:

1. **Shallow Net**: A simple network with minimal layers to observe fundamental learning dynamics.
2. **MLP (Multilayer Perceptron)**: A classic fully-connected neural network with multiple hidden layers.
3. **LeNet-5**: A convolutional network inspired by the LeNet-5 architecture, widely known for its role in early image classification tasks.

## Installation

To run this project, you need Python and the following libraries:
- [PyTorch](https://pytorch.org/get-started/locally/)
- Tensorboard
- Optuna
- ...
Install dependencies with:
```bash
  pip install -r requirement.txt
```

## TensorBoard Logs

The project includes several `runs/` directories that store log data for **TensorBoard**. These logs track model metrics, such as training loss and accuracy, across different training sessions, allowing for a visual analysis of model performance over time.

### Viewing Logs with TensorBoard

To visualize the logs, use TensorBoard. Run the following command, specifying the path to the `runs/` directory:

```bash
tensorboard --logdir=runs/example_run
```
