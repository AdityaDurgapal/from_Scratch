import numpy as np
from Layers.base import Layer
class CNN:
    def __init__(self):
        self.layers=[]
        self.loss_function = None
        self.optimizer = None

    def add_layer(self,layer):
        self.layers.append(layer)

    def forward(self, input_data):
        output=input_data
        for layer in self.layers:
            output=layer.forward(output)
        return output
    
    def backward(self, loss_gradient):
        gradient=loss_gradient
        for layer in reversed(self.layers):
            gradient=layer.backward(gradient)
        
    def train_step(self,x_batch,y_batch):
        predictions = self.forward(x_batch)
        loss=self.loss_function.forward(predictions,y_batch)
        loss_gradient=self.loss_function.backward()
        self.backward(loss_gradient)
        if self.optimizer:
            self.optimizer.update(self.layers)
        
        return loss,predictions
    
    def predict(self, x):
        return self.forward(x)
    
    def compile(self, optimizer, loss_function):
        self.optimizer = optimizer
        self.loss_function = loss_function