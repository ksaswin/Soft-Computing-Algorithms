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
    

    def train_model(self):
        pass


    def predict(self):
        pass


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