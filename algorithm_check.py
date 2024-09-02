import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def load_tensor_from_txt(filename, shape):
    data = np.loadtxt(filename, dtype=np.float32)
    return torch.tensor(data).view(*shape)

batch_size = 1
in_channels = 48
height = 112
width = 112
out_channels = 48
kernel_size = 3
stride = 2
sram_size = 16

quantized_input = load_tensor_from_txt('./pattern/algorithm/quantized_input.txt', (batch_size, in_channels, height, width))

quantized_bias = load_tensor_from_txt('./pattern/algorithm/quantized_bias.txt', (out_channels,))

quantized_weights = load_tensor_from_txt('./pattern/algorithm/quantized_weights.txt', (out_channels, in_channels, kernel_size, kernel_size))


input_slices = []
weight_slices = []
output_slices = []
layer = []
for i in range(0, in_channels, sram_size):
    input_slices.append(quantized_input[:, i : i + sram_size, :, :])
    weight_slices.append(quantized_weights[:, i : i + sram_size, :, :])

golden_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride = stride, padding = 1, bias = None)
golden_layer.weight.data = quantized_weights
golden_output = golden_layer(quantized_input)

for i in range(0, 3):
    layer.append(nn.Conv2d(in_channels = sram_size, out_channels = out_channels, kernel_size = kernel_size, stride = 2, padding=1, bias=None))
    layer[i].weight.data = weight_slices[i]
    output_slices.append(layer[i](input_slices[i]))

final_output = output_slices[0] + output_slices[1] + output_slices[2]

if torch.allclose(final_output, golden_output, rtol=1e-5, atol=1e-8):
    print("Match")
else:
    print("Mismatch")

#Errror calculation
mse = F.mse_loss(final_output, golden_output)
print(f"MSE: {mse.item()}")


max_diff = torch.max(torch.abs(final_output - golden_output))
#print(f"Max absolute difference: {max_diff.item()}")

diff = final_output - golden_output
#print(f"Difference tensor: {diff}")

print(f"{output_slices[0][0][0][0][0].data} + {output_slices[1][0][0][0][0].data} + {output_slices[2][0][0][0][0].data} = {golden_output[0][0][0][0].data} ?")


#Generate pattern
np.savetxt("mem_epoch0_channel0.txt", output_slices[0][0][0].cpu().detach().numpy().flatten(), fmt='%d')
np.savetxt("mem_epoch1_channel0.txt", (output_slices[0][0][0] + output_slices[1][0][0]).cpu().detach().numpy().flatten(), fmt='%d') 
np.savetxt("mem_epoch2_channel0.txt", (output_slices[0][0][0] + output_slices[1][0][0] + output_slices[2][0][0]).cpu().detach().numpy().flatten(), fmt='%d') 