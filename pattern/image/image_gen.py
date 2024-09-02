import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import struct


image_path = '/home/QARepVGG/QARepVGG/data/CIFAR-100_example/apple.png'
image = Image.open(image_path)


transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.PILToTensor(),])

transform_2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

transform_3 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])
image_tensor = transform(image)
image_tensor_norm = transform_2(image)
image_tensor_ = transform_3(image)
print(image_tensor.shape)

channel_R = image_tensor[0, :, :].flatten().to(torch.uint8).numpy()
channel_G = image_tensor[1, :, :].flatten().to(torch.uint8).numpy()
channel_B = image_tensor[2, :, :].flatten().to(torch.uint8).numpy()

channel_R_N = image_tensor_norm[0, :, :].flatten().numpy()
channel_G_N = image_tensor_norm[1, :, :].flatten().numpy()
channel_B_N = image_tensor_norm[2, :, :].flatten().numpy()

channel_R_= image_tensor_[0, :, :].flatten().numpy()
channel_G_ = image_tensor_[1, :, :].flatten().numpy()
channel_B_ = image_tensor_[2, :, :].flatten().numpy()





def write_channel_to_file(channel_data, filename):
    with open(filename, 'w') as f:
        for i in range(0, len(channel_data), 16):
            line_data = channel_data[i:i + 16]
            flatten_input = line_data.tolist()
            formatted_input = []
            for x in flatten_input:
                   formatted_input.append(format(np.uint8(x), '08b'))
            f.write(''.join(formatted_input) + '\n') 

def write_channel_to_file_2(channel_data, filename):
    with open(filename, 'w') as f:
        for value in channel_data:
            f.write(f"{value}\n")

def write_channel_to_file_fp32(channel_data, filename):
    with open(filename, 'w') as f:
        for value in channel_data:
            binary_representation = ''.join(format(byte, '08b') for byte in struct.pack('f', value))
            f.write(binary_representation + '\n')

def write_channel_to_file_fp32_sram_format(channel_data, filename):
    def fp32_to_binary(fp32_value):
        packed = struct.pack('>f', fp32_value)
        int_rep = int.from_bytes(packed, 'big')
        binary_str = f'{int_rep:032b}'
        return binary_str

    with open(filename, 'w') as f:
        for i in range(0, len(channel_data), 16):
            line_data = channel_data[i:i + 16]
            binary_representation = ''.join(
                fp32_to_binary(value) 
                for value in line_data
            )
            f.write(binary_representation + '\n')

def read_fp32_from_file(filename):
    recovered_floats = []
    with open(filename, 'r') as f:
        for line in f:
            for i in range(0, len(line.strip()), 32):
                binary_str = line[i:i + 32]
                byte_data = int(binary_str, 2).to_bytes(4, byteorder='big')
                recovered_floats.append(struct.unpack('f', byte_data)[0])
    return recovered_floats




write_channel_to_file(channel_R, 'image_R.txt')
write_channel_to_file(channel_G, 'image_G.txt')
write_channel_to_file(channel_B, 'image_B.txt')

#write_channel_to_file_2(channel_R_N, 'normalized_R.txt')
#write_channel_to_file_2(channel_G_N, 'normalized_G.txt')
#write_channel_to_file_2(channel_B_N, 'normalized_B.txt')

write_channel_to_file_fp32(channel_R_N, 'normalized_R.txt')
write_channel_to_file_fp32(channel_G_N, 'normalized_G.txt')
write_channel_to_file_fp32(channel_B_N, 'normalized_B.txt')

write_channel_to_file_fp32(channel_R_, 'divided_R.txt')
write_channel_to_file_fp32(channel_G_, 'divided_G.txt')
write_channel_to_file_fp32(channel_B_, 'divided_B.txt')

write_channel_to_file_fp32_sram_format(channel_R_N, 'normalized_R_s.txt')
write_channel_to_file_fp32_sram_format(channel_G_N, 'normalized_G_s.txt')
write_channel_to_file_fp32_sram_format(channel_B_N, 'normalized_B_s.txt')

write_channel_to_file_fp32_sram_format(channel_R_, 'divided_R_s.txt')
write_channel_to_file_fp32_sram_format(channel_G_, 'divided_G_s.txt')
write_channel_to_file_fp32_sram_format(channel_B_, 'divided_B_s.txt')

print(channel_R_[1])
print(channel_R_N[0:16])


recovered_channel_R = read_fp32_from_file('image_R.txt')
recovered_channel_G = read_fp32_from_file('image_G.txt')
recovered_channel_B = read_fp32_from_file('image_B.txt')

def fp32_to_binary(fp32_value):
    packed = struct.pack('>f', fp32_value)
    int_rep = int.from_bytes(packed, 'big')
    binary_str = f'{int_rep:032b}'
    return binary_str


str = fp32_to_binary(1.7253361940383911)
print(str)
print("Success.")
