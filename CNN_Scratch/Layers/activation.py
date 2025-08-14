import numpy as np
import pandas as pd
from .base import Layer

class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.trainable=False
        self.last_input = None

    def forward(self, input_data):
        self.last_input = input_data
        return np.maximum(0,input_data)
    
    def backward(self, prev_gradient):
        return prev_gradient * (self.last_input>0)
    
class Softmax(Layer):
    def __init__(self):
        super().__init__()
        self.last_output= None

    def forward(self, input_data):
        exp_values = np.exp(input_data-np.max(input_data,axis=1, keepdims=True))
        final_value= exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.last_output = final_value
        return final_value

    def backward(self, prev_gradient):
        return prev_gradient
