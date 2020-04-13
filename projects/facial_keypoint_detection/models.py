import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):
    """ Initial model"""
    
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
    
    
    
    
class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()

        # Covolutional Layers
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3)
        self.conv4 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3)

        # Maxpooling Layer
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Dense Layers
        self.fc1 = nn.Linear(in_features = 43264, out_features = 1000)
        self.fc2 = nn.Linear(in_features = 1000,  out_features = 1000)
        self.fc3 = nn.Linear(in_features = 1000,  out_features = 136)

        # Dropouts
        self.drop4 = nn.Dropout(p = 0.4)

        
    def forward(self, x):

        # Convolution + Activation + Pooling
        # Convolution + Activation + Pooling
        # Dropout
        # Convolution + Activation + Pooling 
        # Dropout
        # Convolution + Activation + Pooling 
        
        # Flatten
        
        # Dense + Activation
        # Dense + Activation
        # Dense + Activation
        
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop4(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.drop4(x)
        x = self.pool(F.relu(self.conv4(x)))
        #print(x.shape)

        x = x.view(x.size(0), -1)
        #print(x.shape)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #print(x.shape)


        return x