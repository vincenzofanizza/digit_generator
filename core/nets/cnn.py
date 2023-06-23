'''
Convolutional neural network.

'''
import torch.nn as nn


class Discriminator(nn.Module):
    '''
    Discriminator network.
    
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=320, out_features=50),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=50, out_features=1),
            nn.Sigmoid()  # NOTE: sigmoid ensures the output is between 0 and 1.
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
        
class Generator(nn.Module):
    '''
    Generator network.
    
    '''
    def __init__(self, z_dim):
        super().__init__()
        self.lin1 = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=7*7*64),
            nn.ReLU(),
        )
        self.ct1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.ct2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=7)

    def forward(self, x):
        x = self.lin1(x)
        x = x.view(-1, 64, 7, 7)
        x = self.ct1(x)
        x = self.ct2(x)
        x = self.conv(x)
        return x