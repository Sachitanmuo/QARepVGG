from repvgg import *
import matplotlib.pyplot as plt
import numpy as np

def analysis_tensor(tensor, filename):
    data = tensor.cpu().detach().numpy()
    print(f"Max:{np.max(data)}, Min: {np.min(data)}")
    plt.figure(figsize=(8, 6))
    plt.hist(data.flatten(), bins=50, alpha=0.75, color='b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Histogram')
    plt.savefig(filename)
    plt.show()

device = "cpu"
model = create_QARepVGGBlockV2_A0(deploy=False).to(device)
checkpoint = torch.load('QARepVGGV2-A0testtest/best_checkpoint_v3.pth')
if 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'])
else:
    model.load_state_dict(checkpoint)
model.eval()
switch(model)
#print(model)
inputs = torch.randn(1, 3, 224, 224)
output = model.stage0(inputs)
#kernel = model.linear.state_dict()['weight']
#bias = model.linear.state_dict()['bias']
analysis_tensor(output, "stage0_kernel")
#analysis_tensor(bias, "stage0_bias")




