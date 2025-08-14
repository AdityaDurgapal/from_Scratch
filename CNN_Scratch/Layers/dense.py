import numpy as np
import pandas as pd
from .base import Layer

class Dense(Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.weights = None
        self.biases =  None
        self.last_input =  None

    def initialize_params(self, input_size):
    # Add proper scaling
        self.weights = np.random.randn(input_size, self.units) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, self.units))

    
    def forward(self,input_data):
        if len(input_data.shape)>2:
            batch_size = input_data.shape[0]
            input_data= input_data.reshape(batch_size,-1)
        
        if self.weights is None:
            self.initialize_params(input_data.shape[1])

        self.last_input=input_data
        return np.dot(input_data,self.weights) + self.biases
    
    def backward(self, prev_gradient):
        batch_size = prev_gradient.shape[0]
        self.weights_gradient=np.dot(self.last_input.T,prev_gradient)/batch_size
        self.biases_gradient=np.mean(prev_gradient,axis=0,keepdims=True)
        input_gradient=np.dot(prev_gradient,self.weights.T)
        return input_gradient
    