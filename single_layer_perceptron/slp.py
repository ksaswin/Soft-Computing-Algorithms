from audioop import bias
import pickle
import os

class SingleLayerPerceptron:
    def __init__(self):
        '''Creates a SingleLayerPerceptron model with empty weights and zero bias'''
        self.weights = []
        self.bias = 0

        self.model = {'name': 'Single Layer Perceptron model',
                      'weights': self.weights,
                      'bias': self.bias
                     }
    

    def find_y(self, ynet: float):                                               # y = {1, 0, -1} if ynet {>0, =0, <0}
        if not ynet:
            return 0
        
        if ynet < 0:
            return -1
        
        return 1


    def train_model(self, input_array: list = [], targets: list = [], alpha: float = 1.0, max_epochs: int = 100):
        '''Train the model with an input array, corresponding targets and learning rate.'''
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
        
        for i in range(inputs):
            self.weights.append(0)
        
        x0 = 1

        weight_change = [0 for i in range(inputs)]
        bias_change = 0

        weight_change_flag = [1 for i in range(combinations)]
        epoch = 1

        while(not all(flag == 0 for flag in weight_change_flag)):
            if epoch > max_epochs:
                break

            for i in range(combinations):

                ynet = self.bias
                for j in range(inputs):
                    ynet += input_array[i][j] * self.weights[j]
                
                y = self.find_y(ynet)

                if y != targets[i]:
                    weight_change_flag[i] = 1
                    for j in range(inputs):
                        weight_change[j] = alpha * targets[i] * input_array[i][j]
                        self.weights[j] += weight_change[j]
                    
                    bias_change = alpha * targets[i] * x0
                    self.bias += bias_change
                else:
                    weight_change_flag[i] = 0

                    weight_change = [0 for i in range(inputs)]
                    bias_change = 0
            epoch += 1


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
        
        y = self.find_y(ynet)

        return y


    def load_model(self, model_name: str = 'slp_model'):
        '''Looks for a model named slp_model by default. Specify model name for custom named models.'''
        if os.path.isfile(f'{model_name}.pkl'):
            self.model = pickle.load(open(f'{model_name}.pkl', 'rb'))
            print('Saved model found.')        
        else:
            print(f'Sorry, a model named {model_name} doesnot exist.')


    def save_model(self, model_name: str = 'slp_model'):
        '''Saves the model as slp_model. Specify model name for a custom model name.'''
        pickle.dump(self.model, open(f'{model_name}.pkl', 'wb'))

        print(f'The model was saved as a pickle file, {model_name}.pkl.')