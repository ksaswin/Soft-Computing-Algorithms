import pickle
import os

class HebbNet:
    
    def __init__(self):
        '''Creates a HebbNet model with empty weights and zero bias'''
        self.weights = []
        self.bias = 0

        self.model = {'name': 'Hebb Net model',
                      'weights': self.weights,
                      'bias': self.bias
                     }

    def train_model(self, input_array: list = [], targets: list = []):
        '''Train the model with an input array and corresponding targets.'''
        is_nested = any(isinstance(sub, list) for sub in input_array)

        inputs = len(input_array)
        if is_nested:
            inputs = len(input_array[0])

        combinations = len(targets)
        if len(input_array) != combinations:
            print('Training not possible. Parameters donot match.')
            print(f'Number of input combinations given: {len(self.inputs)}')
            print(f'Number of targets given: {combinations}')
            return

        # Initialise all weights to 0
        for i in range(inputs):
            self.weights.append(0)

        x0 = 1

        weight_change = [0 for i in range(inputs)]
        bias_change = 0

        # Update weights
        for i in range(combinations):
            for j in range(inputs):
                weight_change[j] = input_array[i][j] * targets[i]
                self.weights[j] += weight_change[j]
            
            bias_change = x0 * targets[i]
            self.bias += bias_change


    def predict(self, input_values: list):
        '''Make prediction based on the inputs. Prediction is either 1 or -1.'''
        if len(input_values) != len(self.model['weights']):
            print('Inputs given are not sufficient.')
            print(f"Expected number of inputs: {len(self.model['weights'])}")
            print(f"Given number of inputs: {len(input_values)}")
            return 'no predictions'
        
        ynet = self.model['bias']
        for i in range(len(input_values)):
            ynet += input_values[i] * self.model['weights'][i]
        
        if ynet > 0:
            return 1
        
        return -1


    def load_model(self, model_name : str = 'hebb_model'):
        '''Looks for a model named hebb_model by default. Specify model name for custom named models.'''
        if os.path.isfile(f'{model_name}.pkl'):
            self.model = pickle.load(open(f'{model_name}.pkl', 'rb'))
            print('Saved model found.')        
        else:
            print(f'Sorry, a model named {model_name} doesnot exist.')


    def save_model(self, model_name: str = 'hebb_model'):
        '''Saves the model as hebb_model. Specify model name for a custom model name.'''
        pickle.dump(self.model, open(f'{model_name}.pkl', 'wb'))

        print(f'The model was saved as a pickle file, {model_name}.pkl.')