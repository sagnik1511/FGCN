import torch
import torch.nn as nn
def conv(cin, out, kernel_size = (3, 3), padding = (1, 1), stride = 1):
    return nn.Conv2d(in_channels = cin, out_channels = out, kernel_size = kernel_size, padding = padding, stride = stride)

def first_block(data,  kernel_size = (3,3), padding = (1, 1), stride = 1, pool = "avg"):

    mode1 = nn.Sequential(
            conv(cin = data.shape[1], out = 32, padding = padding, stride = stride),
            nn.BatchNorm2d(num_features = 32, momentum = 0.9),
            nn.ReLU(),
        )
    if pool == 'avg':
        mode2 = nn.Sequential(
            nn.AvgPool2d(kernel_size = kernel_size),
            conv(cin = data.shape[1], out = 32, padding = padding, stride = stride),
            nn.BatchNorm2d(num_features = 32, momentum=0.9),
            nn.ReLU(),
        )
    else:
        mode2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size),
            conv(cin = data.shape[1], out = 32, padding = padding, stride = stride),
            nn.BatchNorm2d(num_features = 32, momentum=0.9),
            nn.ReLU(),
        )
    return mode1(data), mode2(data)

def second_block()


rand_data = torch.rand(1,48,100,100)

out_1, out2 = first_block(rand_data,pool = 'max')
print(out_1.shape, out2.shape)




