import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from repvgg import create_QARepVGGBlockV2_A0
from torchsummary import summary
import time
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
from functions import *
from quantized_repvgg import QuantizedRepVGG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
data_path = './data/'
print(device)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

test_dataset = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

def accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def evaluate(model, q_model, data_loader, device, max_images=100000):
    top1_acc = 0
    top5_acc = 0
    total = 0
    num_images = 0

    top1_acc_ = 0
    top5_acc_ = 0
    total_ = 0
    num_images_ = 0

    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            if num_images >= max_images:
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            quantized_outputs = quantized_model_(model, images)
            #quantized_outputs = q_model(images)
            top1, top5 = accuracy(outputs, labels, topk=(1, 5))
            top1_, top5_ = accuracy(quantized_outputs, labels, topk=(1, 5))
            error_calc(outputs, quantized_outputs)
            to = torch.topk(outputs, 5)
            toq = torch.topk(quantized_outputs, 5)
            print(f"original class:{to}, quantized_class:{toq}")


            top1_acc += top1.item() * images.size(0)
            top5_acc += top5.item() * images.size(0)
            top1_acc_ += top1_.item() * images.size(0)
            top5_acc_ += top5_.item() * images.size(0)
            total += images.size(0)
            total_ += images.size(0)
            num_images += images.size(0)
            num_images_ += images.size(0)

    top1_acc /= total
    top5_acc /= total

    top1_acc_ /= total_
    top5_acc_ /= total_
    return top1_acc, top5_acc, top1_acc_, top5_acc_

def evaluate_quantization(model, data_loader, device, max_images=100):
    top1_acc = 0
    top5_acc = 0
    total = 0
    num_images = 0


    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            if num_images >= max_images:
                break
            images, labels = images.to(device), labels.to(device)
            outputs = quantized_model_(model, images)
            top1, top5 = accuracy(outputs, labels, topk=(1, 5))
            top1_acc += top1.item() * images.size(0)
            top5_acc += top5.item() * images.size(0)
            total += images.size(0)
            num_images += images.size(0)

    top1_acc /= total
    top5_acc /= total

    return top1_acc, top5_acc

def switch(model):
    model.stage0.switch_to_deploy()
    model.stage1[0].switch_to_deploy()
    model.stage1[1].switch_to_deploy()
    model.stage2[0].switch_to_deploy()
    model.stage2[1].switch_to_deploy()
    model.stage2[2].switch_to_deploy()
    model.stage2[3].switch_to_deploy()
    model.stage3[0].switch_to_deploy()
    model.stage3[1].switch_to_deploy()
    model.stage3[2].switch_to_deploy()
    model.stage3[3].switch_to_deploy()
    model.stage3[4].switch_to_deploy()
    model.stage3[5].switch_to_deploy()
    model.stage3[6].switch_to_deploy()
    model.stage3[7].switch_to_deploy()
    model.stage3[8].switch_to_deploy()
    model.stage3[9].switch_to_deploy()
    model.stage3[10].switch_to_deploy()
    model.stage3[11].switch_to_deploy()
    model.stage3[12].switch_to_deploy()
    model.stage3[13].switch_to_deploy()
    model.stage4[0].switch_to_deploy()


def main():
    model = create_QARepVGGBlockV2_A0(deploy=False).to(device)
    checkpoint = torch.load('QARepVGGV2-A0testtest/best_checkpoint.pth')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    switch(model)
    #print(model)
    #quantized_model = QuantizedRepVGG(model)

    #===========Quantized Model Initialize=====================
    # Example usage:
    # Instantiate the original model and the quantized model
    q_ranges = {
        'stage1': [1, 4],
        'stage2': [1, 1, 1, 1],
        'stage3': [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'stage4': [2],
        'linear': 1
    }

    q_model = QuantizedRepVGG(model)

    start_time = time.time()
    top1_acc, top5_acc, top1_acc_, top5_acc_ = evaluate(model, q_model, test_loader, device)
    end_time = time.time()
    #exe_time = end_time - start_time
    print(f'Original  Model - Top-1 Accuracy: {top1_acc:.2f}%, Top-5 Accuracy: {top5_acc:.2f}%')
    print(f'Quantized Model - Top-1 Accuracy: {top1_acc_:.2f}%, Top-5 Accuracy: {top5_acc_:.2f}%')
    '''
    start_time = time.time()
    top1_acc, top5_acc = evaluate_quantization(model, test_loader, device)
    end_time = time.time()
    exe_time = end_time - start_time
    print(f'Original Model - Top-1 Accuracy: {top1_acc:.2f}%, Top-5 Accuracy: {top5_acc:.2f}%, inference time = {exe_time} sec')
    '''
    '''
    quant_modules.initialize()
    #quant_desc_input = QuantDescriptor(num_bits=8, fake_quant=False, calib_method="histogram")
    #quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    #quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    quantized_model = create_QARepVGGBlockV2_A0(deploy=False).to(device)
    checkpoint = torch.load('QARepVGGV2-A0testtest/best_checkpoint.pth')
    if 'model' in checkpoint:
        quantized_model.load_state_dict(checkpoint['model'])
    else:
        quantized_model.load_state_dict(checkpoint)
    quantized_model.eval()
    switch(quantized_model)
    #input_quant_desc = QuantDescriptor(calib_method='histogram')
    #weight_quant_desc = QuantDescriptor(calib_method='histogram', axis=(0,))
    #quant_modules.initialize()
    #quantized_model = QuantizedRepVGG(quantized_model)
    
    
    #print(quantized_model.model.stage1[0].rbr_reparam.state_dict()['bias'][0].dtype)
    # Calibration
    #with torch.no_grad():
    #    for images, _ in test_loader:
    #        images = images.to(device)
    #        quantized_model(images)
    
    #==========Calibration===================
    # Find the TensorQuantizer and enable calibration
    for name, module in quantized_model.named_modules():
        if name.endswith('_quantizer'):
            module.enable_calib()
            module.disable_quant()  # Use full precision data to calibrate
        #else:
        #    module.disable()

    # Feeding data samples
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            quantized_model(images)

    # Finalize calibration
    for name, module in quantized_model.named_modules():
        if name.endswith('_quantizer'):
            module.load_calib_amax()
            module.disable_calib()
            module.enable_quant()
        #else:
        #    module.enable()
    
    quantized_model.cuda()
    #========================================
    #print(quantized_model.stage0.rbr_reparam.state_dict())
    #quantized_model = quant_modules.convert_dequant(quantized_model)
    start_time = time.time()
    top1_acc_quant, top5_acc_quant = evaluate(quantized_model, test_loader, device)
    end_time = time.time()
    exe_time = end_time - start_time
    print(f'Quantized Model - Top-1 Accuracy: {top1_acc_quant:.2f}%, Top-5 Accuracy: {top5_acc_quant:.2f}%, inference time: {exe_time} sec')
    '''
if __name__ == "__main__":
    main()
