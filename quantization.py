import torch
import torch.nn as nn
from repvgg import *
import numpy
import torchvision.transforms as transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
TOP = 3
BOTTOM = -4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

def output_observer(layer, input):
    output = layer(input)
    print(output)

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


def main():
    model = create_QARepVGGBlockV2_A0(deploy= False).to(device)
    switch(model)
    image = cv2.imread("./pattern/image/001.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = Image.fromarray(image)
    
    image = transform(image)
    
    image = image.unsqueeze(0).to(device)
    analysis_tensor(image, "normalized_input")
    model.eval()
    with torch.no_grad():
        output = model.stage0(image)
        analysis_tensor(output, "stage0_output")
        output = model.stage1[0](output)
        analysis_tensor(output, "stage1[0]_output")
        output = model.stage1[1](output)
        analysis_tensor(output, "stage1[1]_output")
        output = model.stage2[0](output)
        analysis_tensor(output, "stage2[0]_output")
        output = model.stage2[1](output)
        analysis_tensor(output, "stage2[1]_output")
        output = model.stage2[2](output)
        analysis_tensor(output, "stage2[2]_output")
        output = model.stage2[3](output)
        analysis_tensor(output, "stage2[3]_output")
        output = model.stage3[0](output)
        analysis_tensor(output, "stage3[0]_output")
        output = model.stage3[1](output)
        analysis_tensor(output, "stage3[1]_output")
        output = model.stage3[2](output)
        analysis_tensor(output, "stage3[2]_output")
        output = model.stage3[3](output)
        analysis_tensor(output, "stage3[3]_output")
        output = model.stage3[4](output)
        analysis_tensor(output, "stage3[4]_output")
        output = model.stage3[5](output)
        analysis_tensor(output, "stage3[5]_output")
        output = model.stage3[6](output)
        analysis_tensor(output, "stage3[6]_output")
        output = model.stage3[7](output)
        analysis_tensor(output, "stage3[7]_output")
        output = model.stage3[8](output)
        analysis_tensor(output, "stage3[8]_output")
        output = model.stage3[9](output)
        analysis_tensor(output, "stage3[9]_output")
        output = model.stage3[10](output)
        analysis_tensor(output, "stage3[10]_output")
        output = model.stage3[11](output)
        analysis_tensor(output, "stage3[11]_output")
        output = model.stage3[12](output)
        analysis_tensor(output, "stage3[12]_output")
        output = model.stage3[13](output)
        analysis_tensor(output, "stage3[13]_output")
        output = model.stage4[0](output)
        analysis_tensor(output, "stage4[0]_output")
        

if __name__ == "__main__":
    main()