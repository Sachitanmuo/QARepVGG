import torch
import torch.nn as nn
from repvgg import *
import numpy
import torchvision.transforms as transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def output_observer(layer, input, input_lowerbound, input_upperbound):
    # Quantize weights and bias
    weights = layer.rbr_reparam.state_dict()['weight']
    bias = layer.rbr_reparam.state_dict().get('bias', None)
    
    quantized_weights, s_weight = quantize_tensor_signed(weights)
    quantized_bias, s_bias = None, None
    if bias is not None:
        quantized_bias, s_bias = quantize_tensor_signed(bias)
    quantized_input, s_input = quantize_tensor_signed(input, input_lowerbound, input_upperbound)

    reconstructed_weights = quantized_weights * s_weight
    if quantized_bias is not None:
        reconstructed_bias = quantized_bias * s_bias
    else:
        reconstructed_bias = None
    output = nn.ReLU()(layer.rbr_reparam(input))
    
    quantized_layer = nn.Conv2d(layer.rbr_reparam.in_channels, layer.rbr_reparam.out_channels, layer.rbr_reparam.kernel_size, 
                                stride=layer.rbr_reparam.stride, padding=layer.rbr_reparam.padding, bias=(reconstructed_bias is not None))
    quantized_layer.weight.data = quantized_weights
    if reconstructed_bias is not None:
        quantized_layer.bias.data = quantized_bias
    
    output_quantized = nn.ReLU()(s_input * s_weight * quantized_layer(quantized_input))
    error_calc(output, output_quantized)
    #print(f"s_input:{s_input}, s_weight:{s_weight}, 2^-5:{2**(-5)}")
    analysis_tensor(output, "unq")
    analysis_tensor(output_quantized, "q")
    return output, output_quantized
'''
def quantize_tensor(tensor, lower_bound, upper_bound):
    scaling_factor = (upper_bound - lower_bound) / 255
    quantized_tensor = torch.clamp(tensor, lower_bound, upper_bound)
    quantized_tensor = 255*(tensor - lower_bound) / (upper_bound - lower_bound)
    quantized_tensor = torch.round(quantized_tensor - 128)
    print("Method 1:")
    print(quantized_tensor)
    return quantized_tensor, scaling_factor
'''
def quantize_tensor_signed(tensor):
    scaling_factor = 1/32.
    quantized_tensor = tensor * 32
    quantized_tensor = torch.clamp(quantized_tensor, -128, 127)
    quantized_tensor = torch.floor(quantized_tensor)
    return quantized_tensor, scaling_factor

def quantize_tensor_unsigned(tensor): #[0, 16] -> [0, 255]
    scaling_factor = 1/16.
    quantized_tensor = tensor * 16
    quantized_tensor = torch.clamp(quantized_tensor, 0, 255)
    quantized_tensor = torch.floor(quantized_tensor)
    return quantized_tensor, scaling_factor

def error_calc(tensor1, tensor2):
    mse_error = torch.mean((tensor1 - tensor2) ** 2)
    abs_error = torch.mean(torch.abs(tensor1 - tensor2))
    print(f"Quantization MSE: {mse_error.item()}")
    print(f"Quantization Absolute Error: {abs_error.item()}")
    analysis_tensor(tensor1, "before_quantize")
    analysis_tensor(tensor2, "after_quantize")
    #print(f"s_input:{s_input}, s_weight:{s_weight}, 2^-5:{2**(-5)}")

def analysis_tensor(tensor, filename):
    data = tensor.cpu().detach().numpy()
    print(f"Max:{np.max(data)}, Min: {np.min(data)}")
    plt.figure(figsize=(8, 6))
    plt.hist(data.flatten(), bins=20, alpha=0.75, color='b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(filename)
    plt.savefig("./plots/" + filename)
    plt.show()
    plt.close()

def quantize_1st_layer(layer, input):
    weights = layer.rbr_reparam.state_dict()['weight']
    bias = layer.rbr_reparam.state_dict().get('bias', None)
    
    quantized_weights, s_weight = quantize_tensor_signed(weights)
    quantized_bias, s_bias = None, None
    if bias is not None:
        quantized_bias, s_bias = quantize_tensor_signed(bias)
    quantized_input, s_input = quantize_tensor_signed(input)

    reconstructed_weights = quantized_weights * s_weight
    if quantized_bias is not None:
        reconstructed_bias = quantized_bias * s_bias
    else:
        reconstructed_bias = None
    output = nn.ReLU()(layer.rbr_reparam(input))
    
    quantized_layer = nn.Conv2d(layer.rbr_reparam.in_channels, layer.rbr_reparam.out_channels, layer.rbr_reparam.kernel_size, 
                                stride=layer.rbr_reparam.stride, padding=layer.rbr_reparam.padding, bias=(reconstructed_bias is not None))
    quantized_layer.weight.data = quantized_weights
    if reconstructed_bias is not None:
        quantized_layer.bias.data = quantized_bias
    
    output_quantized = nn.ReLU()(s_input * s_weight * quantized_layer(quantized_input))
    error_calc(output, output_quantized)
    #print(f"s_input:{s_input}, s_weight:{s_weight}, 2^-5:{2**(-5)}")
    analysis_tensor(output, "unq")
    analysis_tensor(output_quantized, "q")
    return output, output_quantized

def quantize_layer(layer, input):
    weights = layer.rbr_reparam.state_dict()['weight']
    bias = layer.rbr_reparam.state_dict().get('bias', None)
    
    quantized_weights, s_weight = quantize_tensor_signed(weights)
    quantized_bias, s_bias = None, None
    if bias is not None:
        quantized_bias, s_bias = quantize_tensor_signed(bias)
    quantized_input, s_input = quantize_tensor_unsigned(input)

    reconstructed_weights = quantized_weights * s_weight
    if quantized_bias is not None:
        reconstructed_bias = quantized_bias * s_bias
    else:
        reconstructed_bias = None
    output = nn.ReLU()(layer.rbr_reparam(input))
    
    quantized_layer = nn.Conv2d(layer.rbr_reparam.in_channels, layer.rbr_reparam.out_channels, layer.rbr_reparam.kernel_size, 
                                stride=layer.rbr_reparam.stride, padding=layer.rbr_reparam.padding, bias=(reconstructed_bias is not None))
    quantized_layer.weight.data = quantized_weights
    if reconstructed_bias is not None:
        quantized_layer.bias.data = quantized_bias
    
    output_quantized = nn.ReLU()(s_input * s_weight * quantized_layer(quantized_input))
    error_calc(output, output_quantized)
    #print(f"s_input:{s_input}, s_weight:{s_weight}, 2^-5:{2**(-5)}")
    analysis_tensor(output, "unq")
    analysis_tensor(output_quantized, "q")
    return output, output_quantized


