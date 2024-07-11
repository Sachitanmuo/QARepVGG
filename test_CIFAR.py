import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from repvgg import create_QARepVGGBlockV2_A0
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = './data/'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

test_dataset = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)



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

def evaluate(model, data_loader, device):
    top1_acc = 0
    top5_acc = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            top1, top5 = accuracy(outputs, labels, topk=(1, 5))
            top1_acc += top1.item() * images.size(0)
            top5_acc += top5.item() * images.size(0)
            total += images.size(0)

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
    #top1_acc, top5_acc = evaluate(model, test_loader, device)
    #print(f'Original Model - Top-1 Accuracy: {top1_acc:.2f}%, Top-5 Accuracy: {top5_acc:.2f}%')
    
    model_cpu = model.to('cpu')
    model_quantized = torch.quantization.quantize_dynamic(
        model_cpu, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    print(model_quantized.stage0.rbr_reparam.state_dict())

    
    #top1_acc_quant, top5_acc_quant = evaluate(model_quantized, test_loader, 'cpu')
    #print(f'Quantized Model - Top-1 Accuracy: {top1_acc_quant:.2f}%, Top-5 Accuracy: {top5_acc_quant:.2f}%')


if __name__ == "__main__":
    main()