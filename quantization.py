import torch
import torch.nn as nn
from repvgg import *
import numpy
import torchvision.transforms as transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from functions import *
TOP = 3
BOTTOM = -4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])
def main():
    model = create_QARepVGGBlockV2_A0(deploy= False).to(device)
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
    image = images[1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
    image = Image.fromarray(image)   
    image = transform(image)   
    image = image.unsqueeze(0).to(device)
    analysis_tensor(image, "normalized_input")
    model.eval()

    #print(model)
    #input()

    #summ(model)
    #print(quantized_model(model, image).shape)
    #print(model(image).shape)

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
        output = model.gap(output)
        analysis_tensor(output, "average pooling")
        output = output.flatten()
        output = model.linear(output)
        analysis_tensor(output, "linear")
    #input()

    # =========Test the input quantization error=============
    #      input quantization error:0.0003388968762010336
    #========================================================
    #quantized_input, s = quantize_tensor_signed(image)
    #error = torch.sum(torch.mean((image - quantized_input * s)**2))
    #print(f"input quantization error:{error}")


    # =========Test the first layer quantization error=============
    #      Quantization MSE: 0.3397901952266693
    #      Quantization Absolute Error: 0.39588701725006104
    #==============================================================
    print("========== Stage 0 ==========")
    output, output_quantized = quantize_1st_layer(model.stage0, image)
    #error_calc(output, output_quantized)
    #input()
    print("=============================")
    # =========Test the quantization error after ReLU =============
    #      MSE After ReLU: 0.001088797813281417 (use original input)
    #      MSE After ReLU: 0.3483304977416992  (use quantized input)
    #==============================================================
    #q2, s2 = quantize_tensor_unsigned(output_quantized)
    #error = torch.sum(torch.mean((output - q2 * s2)**2))
    #print(f"Quantization Error After ReLU: {error}")
    #input()
    #Stage 1
    print("========== Stage 1 ==========")
    _, output_quantized = quantize_layer(model.stage1[0], output_quantized, q_range_unsigned= 16, q_range_signed= 1, gen_pattern = True)
    output = model.stage1[0](output)
    error_calc(output, output_quantized)
    #input()
    _, output_quantized = quantize_layer(model.stage1[1], output_quantized, q_range_unsigned= 8, q_range_signed= 4, gen_pattern = False)
    output = model.stage1[1](output)
    error_calc(output, output_quantized)
    #input()
    print("=============================")
    
    print("========== Stage 2 ==========")
    q_list = [0.5, 2, 2, 2]
    for i in range (0, 4):
        _, output_quantized = quantize_layer(model.stage2[i], output_quantized, q_range_unsigned= 8, q_range_signed= q_list[i], gen_pattern = False)
        output = model.stage2[i](output)
        error_calc(output, output_quantized)
        #input()
    print("=============================")
    
    print("========== Stage 3 ==========")
    for i in range (0, 14):
        _, output_quantized = quantize_layer(model.stage3[i], output_quantized, q_range_unsigned= 8, q_range_signed= 1, gen_pattern = False)
        output = model.stage3[i](output)
        error_calc(output, output_quantized)
        #input()
    print("=============================")

    #input()
    print("========== Stage 4 ==========")
    _, output_quantized = quantize_layer(model.stage4[0], output_quantized, q_range_unsigned= 8, q_range_signed= 1, gen_pattern = False)
    output = model.stage4[0](output)
    error_calc(output, output_quantized)
    #input()
    print("=============================")

    analysis_tensor(output, "x")
    analysis_tensor(output_quantized, "x")
    print("========== Average Pooling ==========")
    #_, output_quantized = quantize_layer(model.stage4[0], output_quantized)
    output_quantized = torch.round(output_quantized * pow(2, 13))
    output = model.gap(output)
    output_quantized = model.gap(output_quantized)
    output_quantized = output_quantized / pow(2, 13)
    analysis_tensor(output, "x")
    analysis_tensor(output_quantized, "x")
    print("=====================================")
    
    print("========== Linear ===================")
    #_, output_quantized = quantize_layer(model.stage4[0], output_quantized)
    output = torch.flatten(output, 1) 
    output_quantized = torch.flatten(output_quantized, 1) 
    output = model.linear(output)
    #output_quantized = model.linear(output_quantized)
    _, output_quantized = quantize_linear(model.linear, output_quantized, q_range_unsigned=4, q_range_signed=1, gen_pattern=True)
    error_calc(output, output_quantized)
    print("=====================================")

    print("========== Process Output ==========")
    output = torch.topk(output, 5)
    output_quantized = torch.topk(output_quantized, 5)
    print(f"original class:{output}, quantized_class:{output_quantized}")
    

if __name__ == "__main__":
    main()