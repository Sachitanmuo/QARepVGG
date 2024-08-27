import numpy as np

OUT_SIZE = 56
CHANNEL = 48
#To read the golden
with open("./pattern/layer2/output_b.txt", 'r') as f:
    golden = np.zeros((1, CHANNEL, OUT_SIZE, OUT_SIZE))
    
    # Initialize x, y, and channel_index
    x = 0
    y = 0
    channel_index = 0
    x= 0
    y = 0
    for line in f:
        for i in range(0, len(line.strip()), 16):
            binary_str = line.strip()[i:i+16]
            if binary_str[0] == '0':
                number = int(binary_str[1:], 2)
            else:
                number = int(binary_str[1:], 2) - 32768
            
            # Store number in the correct position
            golden[0, i//16, x, y] = number
            #print(f"({i//16},{x},{y})")
        
        # Update y and x
        y += 1
        if y == OUT_SIZE:
            y = 0
            x += 1
print(golden.shape)


#reshape
golden_reshaped = golden.reshape(1, CHANNEL, OUT_SIZE * OUT_SIZE)
#To read the output

with open("./pattern/layer2/output_SRAM_O0.txt", 'r') as f1, open("./pattern/layer2/output_SRAM_O1.txt", 'r') as f2:
    output = np.zeros((1, CHANNEL, OUT_SIZE, OUT_SIZE))
    odd_channel, even_channel = 1, 0
    x, y = 0, 0
    for line in f1:
        if binary_str == 'xxxxxxxxxxxxxxxx':
            binary_str = '0000000000000000'

        try:
            if len(binary_str) != 16:
                raise ValueError(f"Length error: {binary_str}")

            if binary_str[0] == '0':
                number = int(binary_str[1:], 2)
            else:
                number = int(binary_str[1:], 2) - 32768
                

            print(f"(c, x, y, num) = ({even_channel}, {x}, {y}, {number})")
            
        except ValueError as e:
            print(f"Error processing line: {line.strip()} - {str(e)}")
        y += 1
        if(x==OUT_SIZE - 1 and y==OUT_SIZE):
            even_channel += 2
            x = 0
            y = 0
        if y == OUT_SIZE:
            y = 0
            x += 1
    x, y = 0, 0
    for line in f2:
        if binary_str == 'xxxxxxxxxxxxxxxx':
            binary_str = '0000000000000000'

        try:
            if len(binary_str) != 16:
                raise ValueError(f"Length error: {binary_str}")

            if binary_str[0] == '0':
                number = int(binary_str[1:], 2)
            else:
                number = int(binary_str[1:], 2) - 32768
                

            #print(f"(c, x, y, num) = ({odd_channel}, {x}, {y}, {number})")
            
        except ValueError as e:
            print(f"Error processing line: {line.strip()} - {str(e)}")
        y += 1
        if(x==OUT_SIZE - 1 and y==OUT_SIZE):
            odd_channel += 2
            x = 0
            y = 0
        if y == OUT_SIZE:
            y = 0
            x += 1

output_reshaped = output.reshape(1, CHANNEL, OUT_SIZE * OUT_SIZE)

error_threshold = 0
for c in range(4):
    for i in range(OUT_SIZE * OUT_SIZE):
        golden_val = golden_reshaped[0, c, i]
        output_val = output_reshaped[0, c, i]
        error = abs(golden_val - output_val)
        if error > error_threshold:
            x, y = divmod(i, OUT_SIZE)
            print(f"Error at (channel={c}, x={x}, y={y}): Golden={golden_val}, Output={output_val}, Error={error}")

mse = np.mean((golden_reshaped - output_reshaped) ** 2)
print(f"Mean Squared Error (MSE): {mse}")
        


















# golden now contains the restored tensor
#print(golden[0, 0])
'''
Golden_File = "./pattern/layer2/output_b.txt"
Odd_Output_File = "./pattern/layer2/output_b.txt"
Even_Output_File = "./pattern/layer2/output_b.txt"
x = 0
y = 0
with open(Golden_File, 'r') as f1, open(Odd_Output_File, 'r') as f2, open(Even_Output_File, 'r') as f3:
    for line1, line2 in zip(f1, f2):
        for i in range(0, len(line1.strip()), 16):
            binary_str = line1.strip()[i:i+16]
            if binary_str[0] == '0':
                golden_number = int(binary_str[1:], 2)
            else:
                golden_number = int(binary_str[1:], 2) - 32768
            binary_str = line2.strip()[i:i+16]
            if binary_str[0] == '0':
                number = int(binary_str[1:], 2)
            else:
                number = int(binary_str[1:], 2) - 32768
            error = pow((golden_number - number), 2)
            print(f"Pattern [{0}, {i//16}, {x}, {y}] : Golden = {golden_number}, Output = {number}, error = {error}")
        
        y += 1
        if y == OUT_SIZE:
            y = 0
            x += 1
'''