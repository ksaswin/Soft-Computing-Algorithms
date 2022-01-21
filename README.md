## Soft Computing

This repository contains the python implementation for a few soft computing algorithms.

Implemented algorithms:

- Hebb Net
- Single layer perceptron
- Multi layer perceptron (single hidden layer)

### Cloning the repository

Clone the repository using:

```bash
  git clone 'link goes here'
```

Move into the cloned repository:

```bash
  cd Soft-Computing
```

Install the required dependencies:

```bash
  pip install -r requirements.txt
```

### Usage/Examples

Import the packages from different modules using:

```python
from hebbnet.hebb_net import HebbNet
from single_layer_perceptron.slp import SingleLayerPerceptron
from multi_layer_perceptron.mlp import MultiLayerPerceptron
```

Create the model for each algorithm using:

```python
input_list = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
targets = [1, -1, -1, -1]

hebb_model = HebbNet()
slp_model = SingleLayerPerceptron(input_list, targets)
mlp_model = MultiLayerPerceptron(input_list, targets)
```

The example above creates the model for a two input AND gate.
