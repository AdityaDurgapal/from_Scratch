# layers/base.py
import numpy as np

class Layer:
    def __init__(self):
        self.trainable = True
        
    def forward(self, input_data):
        """Forward pass - must be implemented by subclasses"""
        raise NotImplementedError
        
    def backward(self, output_gradient):
        """Backward pass - must be implemented by subclasses"""
        raise NotImplementedError
        
    def get_params(self):
        """Return parameters for optimization"""
        return []
        
    def set_params(self, params):
        """Set parameters from optimizer"""
        pass