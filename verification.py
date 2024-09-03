import numpy as np

OUT_SIZE = 56
CHANNEL = 48


with open("./pattern/layer2/output_b.txt", 'r') as f:
    golden = np.zeros((1, CHANNEL, OUT_SIZE, OUT_SIZE))
    x, y, channel_index = 0, 0, 0
    
    for line in f:
        for i in range(0, len(line.strip()), 16):
            binary_str = line.strip()[i:i+16]
            if binary_str[0] == '0':
                number = int(binary_str[1:], 2)
            else:
                number = int(binary_str[1:], 2) - 32768
            golden[0, channel_index, x, y] = number
            channel_index += 1
            if channel_index == CHANNEL:
                channel_index = 0
                y += 1
                if y == OUT_SIZE:
                    y = 0
                    x += 1
                    if x == OUT_SIZE:
                        x = 0

golden_reshaped = golden.reshape(1, CHANNEL, OUT_SIZE * OUT_SIZE)

#reading output
with open("./pattern/layer2/output_SRAM_O0.txt", 'r') as f1, open("./pattern/layer2/output_SRAM_O1.txt", 'r') as f2:
    output = np.zeros((1, CHANNEL, OUT_SIZE, OUT_SIZE))
    even_channel, odd_channel = 0, 1
    x, y = 0, 0
    
    for line in f1:
        binary_str = line.strip()
        if binary_str == 'xxxxxxxxxxxxxxxx':
            binary_str = '0000000000000000'
        
        if len(binary_str) != 16:
            raise ValueError(f"Length error: {binary_str}")
        
        if binary_str[0] == '0':
            number = int(binary_str[1:], 2)
        else:
            number = int(binary_str[1:], 2) - 32768
        
        output[0, even_channel, y, x] = number
        x += 1
        if x == OUT_SIZE:
            x = 0
            y += 1
            if y == OUT_SIZE:
                x = 0
                y = 0
                even_channel += 2
                
    x, y = 0, 0
    for line in f2:
        binary_str = line.strip()
        if binary_str == 'xxxxxxxxxxxxxxxx':
            binary_str = '0000000000000000'
        
        if len(binary_str) != 16:
            raise ValueError(f"Length error: {binary_str}")
        
        if binary_str[0] == '0':
            number = int(binary_str[1:], 2)
        else:
            number = int(binary_str[1:], 2) - 32768
        
        output[0, odd_channel, x, y] = number
        x += 1
        if x == OUT_SIZE:
            x = 0
            y += 1
            if y == OUT_SIZE:
                x = 0
                y = 0
                odd_channel += 2
        #print(f"(c, y, x, num) = ({odd_channel}, {y}, {x}, {number})")
        #input()

output_reshaped = output.reshape(1, CHANNEL, OUT_SIZE * OUT_SIZE)


error_threshold = 0
for c in range(1):
    for i in range(OUT_SIZE * OUT_SIZE):
        golden_val = golden_reshaped[0, c, i]
        output_val = output_reshaped[0, c, i]
        error = abs(golden_val - output_val)
        y, x = divmod(i, OUT_SIZE)
        #   print(f"channel={c}, y={y}, x={x}): Golden={golden_val}, Output={output_val}, Error={error}")
        #if error > error_threshold:
        #    x, y = divmod(i, OUT_SIZE)
        #    print(f"Error at (channel={c}, x={x}, y={y}): Golden={golden_val}, Output={output_val}, Error={error}")

mse = np.mean((golden_reshaped - output_reshaped) ** 2)
print(f"Mean Squared Error (MSE): {mse}")
