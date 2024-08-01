import torch
import torch.nn as nn
from repvgg import *
import numpy as np
import torchvision.transforms as transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from functions import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

def main():
    model = create_QARepVGGBlockV2_A0(deploy=False).to(device)
    checkpoint = torch.load('QARepVGGV2-A0testtest/best_checkpoint.pth')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    switch(model)

    folder_path = "data/CIFAR-100_example"
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            image = cv2.imread(os.path.join(folder_path, filename))
            if image is not None:
                images.append(image)
    
    layer_max_values = [[] for _ in range(24)]

    with torch.no_grad():
        for img in images:
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = transform(image)
            image = image.unsqueeze(0).to(device)

            layer_outputs = []
            output = model.stage0(image)
            layer_outputs.append(output)

            for i in range(len(model.stage1)):
                output = model.stage1[i](output)
                layer_outputs.append(output)

            for i in range(len(model.stage2)):
                output = model.stage2[i](output)
                layer_outputs.append(output)

            for i in range(len(model.stage3)):
                output = model.stage3[i](output)
                layer_outputs.append(output)

            output = model.stage4[0](output)
            layer_outputs.append(output)

            output = model.gap(output)
            output = output.flatten()
            layer_outputs.append(output)

            output = model.linear(output)
            layer_outputs.append(output)

            for i, layer_output in enumerate(layer_outputs):
                layer_max_values[i].append(layer_output.max().item())

    print("Max values for each layer collected for all images")


    plt.figure(figsize=(20, 10))
    plt.boxplot(layer_max_values)
    plt.xlabel('Layer')
    plt.ylabel('Max Value')
    plt.title('Distribution of Max Values Across Layers')
    plt.savefig('./plots/calibration.png')

if __name__ == "__main__":
    main()
