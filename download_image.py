import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image


transform = transforms.Compose([
    transforms.ToTensor(),
])


cifar100 = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)


classes = cifar100.classes
print("CIFAR-100 Classes:", classes)


output_dir = '/home/QARepVGG/QARepVGG/data/CIFAR-100_example'
os.makedirs(output_dir, exist_ok=True)

saved_classes = {cls: False for cls in classes}

for img, label in cifar100:
    class_name = classes[label]
    if not saved_classes[class_name]:
        img_pil = transforms.ToPILImage()(img)
        img_path = os.path.join(output_dir, f'{class_name}.png')
        img_pil.save(img_path)
        print(f"Saved {class_name} image to {img_path}")
        saved_classes[class_name] = True
    
    if all(saved_classes.values()):
        break

print("All classes have been saved.")
