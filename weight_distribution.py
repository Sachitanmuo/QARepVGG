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

def plot_conv_weights_histogram(model):
    conv_weights = []
    conv_bias = []
    linear_weights = []
    linear_bias = []
    def gather_weights(layer):
        if isinstance(layer, torch.nn.Conv2d) and layer.kernel_size == (3, 3):
            conv_weights.append(layer.weight.data.cpu().numpy().flatten())
            conv_bias.append(layer.bias.data.cpu().numpy().flatten())
        if isinstance(layer, torch.nn.Linear):
            linear_weights.append(layer.weight.data.cpu().numpy().flatten())
            linear_bias.append(layer.bias.data.cpu().numpy().flatten())


    model.apply(gather_weights)


    all_weights = [w for weights in conv_weights for w in weights]
    all_bias = [b for bias in conv_bias for b in bias]

    all_lweights = [w for weights in linear_weights for w in weights]
    all_lbias = [b for bias in linear_bias for b in bias]

    max_weight = max(all_weights)
    min_weight = min(all_weights)
    max_bias = max(all_bias)
    min_bias = min(all_bias)
    max_lweight = max(all_lweights)
    min_lweight = min(all_lweights)
    max_lbias = max(all_lbias)
    min_lbias = min(all_lbias)
    print(f"3*3 convolution weights: max {max_weight}, min {min_weight}")
    print(f"3*3 convolution bias: max {max_bias}, min {min_bias}")
    print(f"Linear weights: max {max_lweight}, min {min_lweight}")
    print(f"Linear bias: max {max_lbias}, min {min_lbias}")
    plt.hist(all_weights, bins=100, edgecolor='black')
    plt.title('Distribution of 3x3 Convolution Weights')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.savefig("conv_weights_distrib")
    plt.clf()

    plt.hist(all_bias, bins=100, edgecolor='black')
    plt.title('Distribution of 3x3 Convolution Bias')
    plt.xlabel('Bias Value')
    plt.ylabel('Frequency')
    plt.savefig("conv_bias_distrib")
    plt.clf()

    plt.hist(all_lweights, bins=100, edgecolor='black')
    plt.title('Distribution of Linear Weights')
    plt.xlabel('Bias Value')
    plt.ylabel('Frequency')
    plt.savefig("linear_weight_distrib")
    plt.clf()

    plt.hist(all_lbias, bins=100, edgecolor='black')
    plt.title('Distribution of Linear Bias')
    plt.xlabel('Bias Value')
    plt.ylabel('Frequency')
    plt.savefig("linear_bias_distrib")
    plt.clf()

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
#analysis_tensor(output, "stage0_kernel")
plot_conv_weights_histogram(model)
#analysis_tensor(bias, "stage0_bias")




