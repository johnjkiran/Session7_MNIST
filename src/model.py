import torch
import torch.nn as nn
import torch.nn.functional as F
# dropout_value = 0.1

class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        # Input Block - Expand
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=0)  # 28x28 -> 26x26 
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 12, kernel_size=3, padding=0)  # 26x26 -> 24x24
        self.bn2 = nn.BatchNorm2d(12)
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14

        # Shrink with 1x1
        self.conv3 = nn.Conv2d(12, 6, 1)  # Stronger channel reduction
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7

        # Second Block - Expand again
        self.conv4 = nn.Conv2d(6, 12, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(12)
        self.conv5 = nn.Conv2d(12, 16, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(16)

        # Maintain high channels for feature extraction
        self.conv6 = nn.Conv2d(16, 14, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(14)
        self.conv7 = nn.Conv2d(14, 12, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(12)

        # Final Block - Shrink to output
        self.conv8 = nn.Conv2d(12, 10, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout1 = nn.Dropout2d(0.02)  # Slightly higher dropout early
        self.dropout2 = nn.Dropout2d(0.01)  # Lower dropout later

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.dropout2(x)
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.dropout2(x)
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.dropout2(x)
        x = self.bn7(F.relu(self.conv7(x)))
        x = self.dropout2(x)
        x = self.conv8(x)
        x = self.avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)