import numpy as np
import pandas as pd
from .base import Layer

class Conv2D(Layer):
    def __init__(self, num_filters, filter_size, stride=1, padding=0):
        super().__init__()
        self.num_filters=num_filters
        self.filter_size= filter_size
        self.stride= stride
        self.padding= padding
        self.weights=None
        self.bias=None
        self.last_input=None
    
    def forward(self, input_data):
        if len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, axis=1)

        batch_size,input_channels,input_height,input_width=input_data.shape

        if self.weights is None:
            self.initialize_param(input_channels)
        
        if self.padding>0:
            input_data= np.pad(input_data,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)))
        
        input_height,input_width=input_data.shape[2:]
        output_height = (input_height-self.filter_size) // self.stride + 1
        output_width = (input_width-self.filter_size) // self.stride + 1
        output = np.zeros((batch_size,self.num_filters,output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                h_start = i*self.stride
                h_end= h_start+ self.filter_size
                w_start = j*self.stride
                w_end= w_start+self.filter_size

                input_slice= input_data[:,:,h_start:h_end,w_start:w_end]

                for filt in range(self.num_filters):
                    output[:,filt,i,j]= np.sum(input_slice*self.weights[filt], axis=(1,2,3)) + self.biases[filt]
        
        self.last_input = input_data
        return output

    def initialize_param(self, input_channels):
        # Use He initialization for ReLU networks
        fan_in = input_channels * self.filter_size * self.filter_size
        self.weights = np.random.randn(
            self.num_filters, input_channels, 
            self.filter_size, self.filter_size
        ) * np.sqrt(2.0 / fan_in)
        self.biases = np.zeros((self.num_filters, 1))


    def backward(self, prev_gradient):
        batch_size, _, out_height, out_width = prev_gradient.shape
        _, input_channels, padded_height, padded_width= self.last_input.shape
        weights_gradient=np.zeros_like(self.weights)
        biases_gradient=np.zeros_like(self.biases)
        input_gradient=np.zeros_like(self.last_input)

        for i in range(out_height):
            for j in range(out_width):
                h_start=i*self.stride
                h_end=h_start+self.filter_size
                w_start=j*self.stride
                w_end=w_start+self.filter_size

                input_slice=self.last_input[:,:, h_start:h_end, w_start:w_end]

                for filt in range(self.num_filters):
                    weights_gradient[filt] += np.sum(input_slice*prev_gradient[:,filt:filt+1,i:i+1,j:j+1], axis=0)
                    input_gradient[:,:,h_start:h_end,w_start:w_end]+=(self.weights[filt]*prev_gradient[:,filt:filt+1,i:i+1,j:j+1])
                    biases_gradient[filt]+=np.sum(prev_gradient[:,filt,i,j])
        self.weights_gradient=weights_gradient
        self.biases_gradient=biases_gradient

        if self.padding>0:
            input_gradient=input_gradient[:,:,self.padding:-self.padding,self.padding:-self.padding]
        
        return input_gradient
