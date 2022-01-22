## Single Layer Perceptron

This repository contains the implementation of Single Layer perceptron network using python.

The methods you can call from the class:

- train_model()
- predict()
- load_model()
- save_model()

### Usage/Examples

The SingleLayerPerceptron class creates a model with empty weights and zero bias.

The train_model() method takes two lists as parameters for training the model.

The first list takes all the input combinations for training the model and the second list takes the corresponding targets for each of the combinations. Other optional arguments in the method are alpha and max_epochs. alpha is the learning rate and by default the learning rate is set to 1.0 and max_epochs is the iteration limiter which by default is 100.

The input list is a list of lists: [[x1, x2], [x1, x2]]

The target list is the targets for each of the combinations above: [t1, t2]

alpha expects a float value: alpha = 0.5

max_epochs expects an int value: max_epochs = 50

The example below shows a code snipped that implements a two input AND gate.

```python
from single_layer_perceptron.slp import SingleLayerPerceptron

model = SingleLayerPerceptron()

input_list = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
target_values = [1, -1, -1, -1]
model.train_model(input_list, target_values, alpha=1, max_epochs=3)

model.save_model()
```

save_model() method can also take a string parameter to save the model in any name you wish to. By default, the model name will be slp_model.pkl

```python
model.save_model('MyModelName)
```

This will save the model as MyModelName.

To load a model that was already saved and to perform a prediction, follow the example below.

```python
from single_layer_perceptron.slp import SingleLayerPerceptron

model = SingleLayerPerceptron()
mode.load_model()

prediction_inputs = [1, 1]
prediction = model.predict(prediction_inputs)

print(prediction)
```

The predicted values will be either 1 or -1.

The parameters required in the predict() method is a list of inputs to perform the prediction.

The load_model() method, by default looks for a model named slp_model. TO look for a specific model name, add the model name within the paranthesis.

```python
model.load_model('MyModelName')
```
