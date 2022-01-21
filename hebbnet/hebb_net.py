import pickle
import os

class HebbNet:
    
    def __init__(self, input_array: list = [], targets: list = []):
        self.input_array = input_array
        self.targets = targets

        is_nested = any(isinstance(sub, list) for sub in self.input_array)

        self.inputs = len(self.input_array)
        if is_nested:
            self.inputs = len(self.input_array[0])

        self.combinations = len(self.targets)

        self.weights = [0 for i in range(self.inputs)]
        self.bias = 0

        self.model = {'name': 'Hebb Net model',
                      'weights': self.weights,
                      'bias': self.bias}

    def train_model(self):
        if len(self.input_array) != self.combinations:
            print('Training not possible. Parameters donot match.')
            print(f'Number of input combinations given: {len(self.inputs)}')
            print(f'Number of targets given: {self.combinations}')
            return

        self.bias = 0
        self.x0 = 1

        self.weight_change = [0 for i in range(self.inputs)]
        self.bias_change = 0

        for i in range(self.combinations):
            for j in range(self.inputs):
                self.weight_change[j] = self.input_array[i][j] * self.targets[i]
                self.weights[j] += self.weight_change[j]
            
            self.bias_change = self.x0 * self.targets[i]
            self.bias += self.bias_change


    def predict(self, input_values: list):
        if len(input_values) != len(self.model['weights']):
            print('Inputs given are not sufficient.')
            print(f"Expected number of inputs: {len(self.model['weights'])}")
            print(f"Given number of inputs: {len(input_values)}")
        
        ynet = self.model['bias']
        for i in range(len(input_values)):
            ynet += input_values[i] * self.model['weights'][i]
        
        if ynet > 0:
            return 1
        
        return -1


    def load_model(self, model_name : str = 'hebb_model'):
        if os.path.isfile(f'{model_name}.pkl'):
            self.model = pickle.load(open(f'{model_name}.pkl', 'rb'))
        else:
            print(f'Sorry, a model named {model_name} doesnot exist.')


    def save_model(self, model_name: str = 'hebb_model'):
        pickle.dump(self.model, open(f'{model_name}.pkl', 'wb'))

        print(f'The model was saved as a pickle file, {model_name}.pkl.')