import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, size_block, in_channel=1):
        super(ConvNet, self).__init__()

        self.size_block = size_block
        out_channel = 2

        self.conv1 = nn.Conv2d(in_channel, out_channel, 7, padding=3, stride=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 7, padding=3, stride=1)
        self.conv3 = nn.Conv2d(out_channel, in_channel, 7, padding=3, stride=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.sigmoid(x)

        return x
