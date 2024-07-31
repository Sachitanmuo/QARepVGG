import torch
import torch.nn as nn
from repvgg import *
import numpy
import torchvision.transforms as transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
def flatten_and_format(state_dict):
    state_dict = state_dict.cpu().detach().numpy().astype(np.int16)
    return ' '.join(map(str, state_dict.flatten().tolist()))

def flatten_and_format_binary(data, data_type):
    data = data.cpu().detach().numpy()
    if data_type == 'uint8':
        data = data.astype(np.uint8)
        return ' '.join(format(x, '08b') for x in data.flatten())
    elif data_type == 'int8':
        data = data.astype(np.int8)
        formatted_data = []
        for x in data.flatten():
            if x >= 0:
                formatted_data.append('0' + format(np.int8(x), '07b'))
            else:
                formatted_data.append('1' + format(np.int8(128 + x), '07b'))
        return ' '.join(formatted_data)
    elif data_type == 'int16':
        data = data.astype(np.int16)
        formatted_data = []
        for x in data.flatten():
            if x >= 0:
                formatted_data.append('0' + format(np.int16(x), '015b'))
            else:
                formatted_data.append('1' + format(np.int16(32768 + x), '015b'))
        return ' '.join(formatted_data)
    
def generate_pattern(weight, bias, input, output):
    os.makedirs("./pattern/layer2/", exist_ok=True)
    #analysis_tensor(weight, "l2qw")
    weight = weight.cpu().detach().numpy()
    bias   =   bias.cpu().detach().numpy()
    input  =  input.cpu().detach().numpy()
    output = output.cpu().detach().numpy()

    out_channels, in_channels, kernel_h, kernel_w = weight.shape
    in_size = input.shape[2]
    out_size = output.shape[2]
    with open("./pattern/layer2/weight.txt", 'w') as f:
        for i in range (out_channels):
            for j in range(in_channels):
                flatten_weight = weight[i, j].flatten().astype(np.int8).tolist()
                data = ' '.join(map(str, flatten_weight))
                f.write(data + '\n')
    
    with open("./pattern/layer2/weight_b.txt", 'w') as f:
        for i in range (out_channels):
            for j in range(in_channels):
                flatten_weight = weight[i, j].flatten().tolist()
                formatted_weight = []
                for x in flatten_weight:
                    if x >= 0:
                        formatted_weight.append('0' + format(np.int8(x), '07b'))
                    else:
                        formatted_weight.append('1' + format(np.int8(x + 128), '07b'))
                data = ' '.join(map(str, formatted_weight))
                f.write(data + '\n')
            
    with open("./pattern/layer2/bias.txt", 'w') as f:
        flatten_bias = bias.flatten().astype(np.int8).tolist()
        data = ' '.join(map(str, flatten_bias))
        f.write(data + '\n')
    
    with open("./pattern/layer2/bias_b.txt", 'w') as f:
        flatten_bias = bias.flatten().astype(np.int8).tolist()
        formatted_bias = []
        for x in flatten_bias:
            if x >= 0:
                formatted_bias.append('0' + format(np.int8(x), '07b'))
            else:
                formatted_bias.append('1' + format(np.int8(x + 128), '07b'))
        data = ' '.join(map(str, formatted_bias))
        f.write(data + '\n')

    with open("./pattern/layer2/output.txt", 'w') as f:
        for i in range (out_size):
            for j in range (out_size):
                flatten_output = output[:, :, i, j].flatten().astype(np.int16).tolist()
                data = ' '.join(map(str, flatten_output))
                f.write(data + '\n')

    with open("./pattern/layer2/output_b.txt", 'w') as f:
        for i in range (out_size):
            for j in range (out_size):
                flatten_output = output[:, :, i, j].flatten().astype(np.int16).tolist()
                formatted_output = []
                for x in flatten_output:
                    if x >= 0:
                        formatted_output.append('0' + format(np.int16(x), '015b'))
                    else:
                        formatted_output.append('1' + format(np.int16(x + 32768), '015b'))
                data = ' '.join(map(str, formatted_output))
                f.write(data + '\n')
    #Seperate the channel into 3 parts
    input_split = []
    for i in range (3):
        input_split.append(input[:, i: 16*(i+1), :, :])
    h0_w0 = input_split[0][:, :, 0::2, 0::2]
    h0_w1 = input_split[0][:, :, 0::2, 1::2]
    h1_w0 = input_split[0][:, :, 1::2, 0::2]
    h1_w1 = input_split[0][:, :, 1::2, 1::2]
    def repeat(tensor, file):
        with open(file, 'w') as f:
            for i in range (56):
                for j in range (56):
                    flatten_input = tensor[:, :, i, j].flatten().astype(np.uint8).tolist()
                    formatted_input = []
                    for x in flatten_input:
                        formatted_input.append(format(np.uint8(x), '09b'))
                    data = ' '.join(map(str, formatted_input))
                    f.write(data + '\n')
    repeat(h0_w0, "./pattern/layer2/input_cgroup0h%0w%0.txt")
    repeat(h0_w1, "./pattern/layer2/input_cgroup0h%0w%1.txt")
    repeat(h1_w0, "./pattern/layer2/input_cgroup0h%1w%0.txt")
    repeat(h1_w1, "./pattern/layer2/input_cgroup0h%1w%1.txt")
    
    h0_w0 = input_split[1][:, :, 0::2, 0::2]
    h0_w1 = input_split[1][:, :, 0::2, 1::2]
    h1_w0 = input_split[1][:, :, 1::2, 0::2]
    h1_w1 = input_split[1][:, :, 1::2, 1::2]

    repeat(h0_w0, "./pattern/layer2/input_cgroup1h%0w%0.txt")
    repeat(h0_w1, "./pattern/layer2/input_cgroup1h%0w%1.txt")
    repeat(h1_w0, "./pattern/layer2/input_cgroup1h%1w%0.txt")
    repeat(h1_w1, "./pattern/layer2/input_cgroup1h%1w%1.txt")

    h0_w0 = input_split[2][:, :, 0::2, 0::2]
    h0_w1 = input_split[2][:, :, 0::2, 1::2]
    h1_w0 = input_split[2][:, :, 1::2, 0::2]
    h1_w1 = input_split[2][:, :, 1::2, 1::2]

    repeat(h0_w0, "./pattern/layer2/input_cgroup2h%0w%0.txt")
    repeat(h0_w1, "./pattern/layer2/input_cgroup2h%0w%1.txt")
    repeat(h1_w0, "./pattern/layer2/input_cgroup2h%1w%0.txt")
    repeat(h1_w1, "./pattern/layer2/input_cgroup2h%1w%1.txt")

    



    


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
def quantize_tensor_signed(tensor, q_range):
    scaling_factor = 2*q_range / 256.
    quantized_tensor = tensor / scaling_factor
    quantized_tensor = torch.clamp(quantized_tensor, -128, 127)
    quantized_tensor = torch.round(quantized_tensor)
    return quantized_tensor, scaling_factor

def quantize_tensor_unsigned(tensor, q_range): #[0, 16] -> [0, 255]
    scaling_factor = q_range / 256
    quantized_tensor = tensor * (256/q_range)
    quantized_tensor = torch.clamp(quantized_tensor, 0, 255)
    quantized_tensor = torch.round(quantized_tensor)
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
    
    quantized_weights, s_weight = quantize_tensor_signed(weights, 4)
    quantized_bias, s_bias = None, None
    if bias is not None:
        quantized_bias, s_bias = quantize_tensor_signed(bias, 4)
    quantized_input, s_input = quantize_tensor_signed(input, 4)

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

def quantize_layer(layer, input, q_range_unsigned, q_range_signed, gen_pattern = False):
    weights = layer.rbr_reparam.state_dict()['weight']
    bias = layer.rbr_reparam.state_dict().get('bias', None)
    
    quantized_weights, s_weight = quantize_tensor_signed(weights, q_range_signed)
    quantized_bias, s_bias = None, None
    if bias is not None:
        quantized_bias, s_bias = quantize_tensor_signed(bias, 4)
    quantized_input, s_input = quantize_tensor_unsigned(input, q_range_unsigned)

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
    o = nn.ReLU()(quantized_layer(quantized_input))
    error_calc(output, output_quantized)
    #print(f"s_input:{s_input}, s_weight:{s_weight}, 2^-5:{2**(-5)}")
    analysis_tensor(output, "unq")
    analysis_tensor(output_quantized, "q")

    #=================== generate pattern ====================
    print(quantized_weights.shape)
    print(quantized_bias.shape)
    print(quantized_input.shape)
    print(o.shape)
    analysis_tensor(o, "ooooo")
    analysis_tensor(bias, "ori")
    analysis_tensor(quantized_bias, "l2qb")
    if gen_pattern:
        generate_pattern(quantized_weights, quantized_bias, quantized_input, o)
    #==========================================================
    return output, output_quantized


