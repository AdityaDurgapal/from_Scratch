import numpy as np
from .base import Layer

class MaxPool2D(Layer):
    def __init__(self, pool_size=2, stride=2):
        super().__init__()
        self.pool_size=pool_size
        self.stride= stride
        self.last_input=None

    def forward(self, input_data):
        batch_size, channels, input_height, input_width=input_data.shape
        self.mask=np.zeros_like(input_data)
        output_height = (input_height-self.pool_size) // self.stride + 1
        output_width = (input_width-self.pool_size) // self.stride + 1
        output = np.zeros((batch_size,channels,output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                h_start=i*self.stride
                h_end= h_start+self.pool_size
                w_start= j*self.stride
                w_end=w_start+self.pool_size

                input_slice= input_data[:,:,h_start:h_end,w_start:w_end]
                input_slice_reshaped= input_slice.reshape(batch_size,channels,-1)
                max_indices=np.argmax(input_slice_reshaped, axis=2)
                output[:,:,i,j]=np.max(input_slice_reshaped, axis=2)

                for b in range(batch_size):
                    for c in range(channels):
                        max_index=max_indices[b,c]
                        max_h = h_start+max_index//self.pool_size
                        max_w = w_start+max_index%self.pool_size
                        self.mask[b,c,max_h,max_w] =1
        
        self.last_input=input_data
        return output

    def backward(self, prev_gradient):
        if len(prev_gradient.shape) == 2:
            batch_size = prev_gradient.shape[0]
            channels = self.last_input.shape[1]
            input_height, input_width = self.last_input.shape[2:]
            output_height = (input_height - self.pool_size) // self.stride + 1
            output_width = (input_width - self.pool_size) // self.stride + 1
            prev_gradient = prev_gradient.reshape(batch_size, channels, output_height, output_width)
        
        batch_size, channels, output_height, output_width = prev_gradient.shape
        input_gradient = np.zeros_like(self.last_input)

        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                input_gradient[:, :, h_start:h_end, w_start:w_end] += (
                    self.mask[:, :, h_start:h_end, w_start:w_end] * 
                    prev_gradient[:, :, i:i+1, j:j+1]
                )
        
        return input_gradient

