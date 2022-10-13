import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


### https://github.com/locuslab/TCN/blob/master/TCN/tcn.py


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, input_size, output_size, num_blocks, kernel_size=2, dropout=0.2):
        ## num_blocks = [a, b, c, d, e], 5 blocks each with size of a, b, c, d, e
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_blocks)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_size = input_size if i == 0 else num_blocks[i - 1]
            out_size = num_blocks[i]
            layers += [TemporalBlock(in_size, out_size, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.bilinear = nn.Bilinear(output_size, output_size, 128)
        self.fc = nn.Linear(128, output_size)
        self.soft = nn.Softmax(0)

    def forward(self, below, above):
        below_feature = self.network(below)
        above_feature = self.network(above)
        out = self.bilinear(below_feature, above_feature)
        out = self.fc(out)
        out = self.soft(out)
        return out

    # def forward(self, below):
    #     below_feature = self.network(below)
    #     out = self.bilinear(below_feature)
    #     out = self.fc(out)
    #     out = self.soft(out)
    #     return out



