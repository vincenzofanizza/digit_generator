'''
Fully connected network.

'''
import torch.nn as nn


class Discriminator(nn.Module):
    '''
    Discriminator network. 
    
    '''
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=128),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=128, out_features=1), 
            nn.Sigmoid()  # NOTE: sigmoid ensures the output is between 0 and 1.
        )
    
    def forward(self, x):
        return self.disc(x)
    
class Generator(nn.Module):
    '''
    Generator network.
    
    '''
    def __init__(self, z_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=256),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=256, out_features=28*28), 
            nn.Tanh()  # NOTE: tanh ensures the output is between -1 and 1, just like the normalized MNIST images.
        )

    def forward(self, x):
        return self.gen(x)
