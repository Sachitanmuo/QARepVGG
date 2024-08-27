import torch
from PIL import Image
from torchvision import transforms
import numpy as np

image_path = '/home/QARepVGG/QARepVGG/data/CIFAR-100_example/apple.png'
image = Image.open(image_path)


transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.PILToTensor(),])

image_tensor = transform(image)
print(image_tensor.shape)

channel_R = image_tensor[0, :, :].flatten().to(torch.uint8).numpy()
channel_G = image_tensor[1, :, :].flatten().to(torch.uint8).numpy()
channel_B = image_tensor[2, :, :].flatten().to(torch.uint8).numpy()

def write_channel_to_file(channel_data, filename):
    with open(filename, 'w') as f:
        for i in range(0, len(channel_data), 16):
            line_data = channel_data[i:i + 16]
            flatten_input = line_data.tolist()
            formatted_input = []
            for x in flatten_input:
                   formatted_input.append(format(np.uint8(x), '08b'))
            f.write(''.join(formatted_input) + '\n') 

write_channel_to_file(channel_R, 'image_R.txt')
write_channel_to_file(channel_G, 'image_G.txt')
write_channel_to_file(channel_B, 'image_B.txt')
print("Success.")
