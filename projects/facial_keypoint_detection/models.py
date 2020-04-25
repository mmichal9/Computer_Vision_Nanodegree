import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
    
    
    
class NetTiny(nn.Module):
    """ Initial model"""
    
    def __init__(self):
        """ """
        super(NetTiny, self).__init__()
        self.conv = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        self.fc = nn.Linear(in_features = 484128, out_features = 68*2)
    
    def forward(self, x):
        """ """
        x = F.relu(F.max_pool2d(self.conv(x),2))
        x = x.view(x.size(0), -1)
        x = self.fc(x) # last layer no activation!
        return x

    
    
    
class KeyPointNetCNN(nn.Module):

    def __init__(self):
        super(KeyPointNetCNN, self).__init__()
        """ """
        # Covolutional:
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3)
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 96, kernel_size = 3)

        # Batch Normalisation CNN:
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4_bn = nn.BatchNorm2d(96)
        
        # Batch Normalisation Dens:
        self.fc1_bn = nn.BatchNorm1d(6528)
        self.fc2_bn = nn.BatchNorm1d(1088)
        
        # Dense:
        self.fc1 = nn.Linear(in_features = 16224, out_features = 6528)
        self.fc2 = nn.Linear(in_features = 6528, out_features = 1088)
        self.fc3 = nn.Linear(in_features = 1088, out_features = 68*2)

        # Dropouts:
        self.drop1 = nn.Dropout(p = 0.1)
        self.drop2 = nn.Dropout(p = 0.2)
        self.drop4 = nn.Dropout(p = 0.4)

        
    def forward(self, x):
        """ """
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)),2))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)),2))
        x = self.drop1(x)
        
        x = F.relu(F.max_pool2d(self.conv3_bn(self.conv3(x)),2))
        x = self.drop2(x)
        
        x = F.relu(F.max_pool2d(self.conv4_bn(self.conv4(x)),2))
        x = self.drop2(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.drop2(F.relu(self.fc1_bn(self.fc1(x))))
        x = self.drop2(F.relu(self.fc2_bn(self.fc2(x))))
        x = self.fc3(x) # last layer no activation!
        return x
    
    