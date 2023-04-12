from HSI_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

'''
Creates a Convolutional Encoder object.
- n_channels = number of spectral channels of the HSI input image.
- zdim = number of characteristics of the output latent vector
- spatial_dim = spatial X and Y dimmensions (square spatial shape) of the input image
- verbose = enables debugging traces
'''
class Encoder(torch.nn.Module):
    def __init__(self, n_channels = 207, zdim = 512, spatial_dim = 100, verbose = 0):
        super(Encoder, self).__init__()
        self.n_channels = n_channels
        self.zdim = zdim
        self.spatial_dim = spatial_dim
        self.verbose = verbose

        # Feature extraction
        self.conv0 = nn.Conv2d(self.n_channels, 128, kernel_size=1, stride=1, padding=0) #100x100
        self.norm0 = nn.BatchNorm2d(128)
        #gelu

        self.conv1 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1) #50x50
        self.norm1 = nn.BatchNorm2d(128)
        #gelu

        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) #50x50
        self.norm2 = nn.BatchNorm2d(128)
        #gelu
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) #50x50
        self.norm3 = nn.BatchNorm2d(128)
        # Skip connection + gelu

        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) #25x25
        self.norm4 = nn.BatchNorm2d(256)
        #gelu
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) #13x13
        self.norm5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1) #7x7
        self.norm6 = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1) #4x4
        self.norm7 = nn.BatchNorm2d(1024)

        self.conv8 = nn.Conv2d(1024, 2048, kernel_size=2, stride=2, padding=0) #2x2
        self.norm8 = nn.BatchNorm2d(2048)
        # Skip connection + gelu

        # MLP bottleneck and latent space
        self.fc_mu = nn.Linear(2048*2*2, self.zdim)
        self.fc_logvar = nn.Linear(2048*2*2, self.zdim)

    # Parametrization trick to allow the backpropagation of the network
    def reparameterize(self, mu, log_var):
        # std = torch.exp(0.5 * log_var)  # standard deviation
        std = log_var.mul(0.5).exp_()
        eps = torch.randn_like(std)
        # eps = torch.randn(*mu.size())

        sample = mu + (eps * std)  # sampling
        return sample

    # Forward propagation
    def forward(self, x):
        self.act_maps = []
        if self.verbose == 1: print(f'>> Input data shape = {torch.Tensor.size(x)}')

        # Feature extraction
        x0 = self.conv0(x)
        x0 = F.gelu(x0)
        if self.verbose == 1: print(f'Activation map 0 dimmensions (conv) = {torch.Tensor.size(x0)}')
        self.act_maps.append(x0)

        x1 = self.conv1(x0)
        # x1 = self.norm1(x1)
        x1 = F.gelu(x1)
        if self.verbose == 1: print(f'Activation map 1 dimmensions (conv) = {torch.Tensor.size(x1)}')
        self.act_maps.append(x1)

        x2 = self.conv2(x1)
        # x2 = self.norm2(x2)
        x2 = F.gelu(x2)
        if self.verbose == 1: print(f'Activation map 2 dimmensions (conv) = {torch.Tensor.size(x2)}')
        self.act_maps.append(x2)

        x3 = self.conv3(x2)
        # x3 = self.norm3(x3)
        x3 = F.gelu(x3+x1)  # Skip connection
        if self.verbose == 1: print(f'Activation map 3 dimmensions (conv) = {torch.Tensor.size(x3)}')
        self.act_maps.append(x3)

        x4 = self.conv4(x3)
        # x4 = self.norm4(x4)
        x4 = F.gelu(x4)
        if self.verbose == 1: print(f'Activation map 4 dimmensions (conv) = {torch.Tensor.size(x4)}')
        self.act_maps.append(x4)

        x5 = self.conv5(x4)
        # x5 = self.norm5(x5)
        # x5 = F.gelu(x5+x4)  # Skip connection
        x5 = F.gelu(x5)  # Skip connection
        if self.verbose == 1: print(f'Activation map 5 dimmensions (conv) = {torch.Tensor.size(x5)}')
        self.act_maps.append(x5)

        x6 = self.conv6(x5)
        x6 = F.gelu(x6) 
        if self.verbose == 1: print(f'Activation map 6 dimmensions (conv) = {torch.Tensor.size(x6)}')
        self.act_maps.append(x6)

        x7 = self.conv7(x6)
        x7 = F.gelu(x7) 
        if self.verbose == 1: print(f'Activation map 7 dimmensions (conv) = {torch.Tensor.size(x7)}')
        self.act_maps.append(x7)

        x8 = self.conv8(x7)
        x8 = F.gelu(x8) 
        if self.verbose == 1: print(f'Activation map 6 dimmensions (conv) = {torch.Tensor.size(x8)}')
        self.act_maps.append(x8)

        # Bottleneck latent vector
        x_end = x8.view(x8.size(0), -1)
        if self.verbose == 1: print(f'Reshaped dimmensions (flatten) = {torch.Tensor.size(x_end)}')
        # x = F.gelu(self.fc1(x))
        # if self.verbose == 1: print(f'Dimmensions of the linear layer 1 = {torch.Tensor.size(x)}')

        z_mu = self.fc_mu(x_end)            # Without activation function.
        z_logvar = self.fc_logvar(x_end)    # Without activation function.
        z = self.reparameterize(z_mu, z_logvar)
        if self.verbose == 1: print(f'Dimmensions of the output latent vector = {torch.Tensor.size(z)}')

        return z, z_mu, z_logvar, self.act_maps

################################################

'''
Creates a Convolutional Decoder object.
- zdim = number of characteristics of the input latent vector
- n_channels = number of spectral channels of the HSI output image.
- spatial_dim = spatial X and Y dimmensions (square spatial shape) of the output image
- verbose = enables debugging traces
'''
class Decoder(torch.nn.Module):

    def __init__(self, zdim=512, n_channels=207, spatial_dim=100, verbose = 0):
        super(Decoder, self).__init__()
        self.spatial_dim = spatial_dim
        self.n_channels = n_channels
        self.verbose = verbose

        ## 1) Latent space and MLP bottleneck
        self.fc_z = nn.Linear(zdim, 2048*2*2) #2x2

        ## 2) Feature reconstruction
        self.trans_conv000 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2, padding=0) #4x4
        self.trans_conv00 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1) #7x7

        self.trans_conv0 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1) #13x13
        self.trans_conv05 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1) #13x13
        self.trans_conv1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1)  #13x13
        self.norm1 = nn.BatchNorm2d(256)
        #gelu
        self.trans_conv2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1) #25x25
        self.norm2 = nn.BatchNorm2d(256)
        # Skip connection + gelu
        self.trans_conv23 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1)  #25x25
        self.trans_conv27 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1)  #25x25
        
        self.trans_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) #50x50
        self.norm3 = nn.BatchNorm2d(128)
        #gelu
        self.trans_conv4 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1) #50x50
        self.norm4 = nn.BatchNorm2d(128)
        # Skip connection + gelu

        self.trans_conv5 = nn.ConvTranspose2d(128, self.n_channels, kernel_size=4, stride=2, padding=1) #100x100
        self.norm5 = nn.BatchNorm2d(self.n_channels)
        #gelu


    def forward(self, x):
        if self.verbose == 1: print(f'\n>> Input latent vector shape: {torch.Tensor.size(x)}')
        self.act_maps = []

        x = self.fc_z(x)
        if self.verbose == 1: print(f'Output dense layer dimmensions = {torch.Tensor.size(x)}')
        x = x.view(x.size(0), 2048, 2, 2)      #2x2

        if self.verbose == 1: print(f'Reshaped dimmensions = {torch.Tensor.size(x)}')
        self.act_maps.append(x)

        x000 = self.trans_conv000(x)  #4x4
        x000 = F.gelu(x000)
        if self.verbose == 1: print(f'Activation map 000 dimmensions (conv) = {torch.Tensor.size(x000)}')
        self.act_maps.append(x000)

        x00 = self.trans_conv00(x000)        #7x7
        x00 = F.gelu(x00)
        if self.verbose == 1: print(f'Activation map 00 dimmensions (conv) = {torch.Tensor.size(x00)}')
        self.act_maps.append(x00)

        x0 = self.trans_conv0(x00)            #13x13
        x0 = F.gelu(x0)
        if self.verbose == 1: print(f'Activation map 0 dimmensions (conv) = {torch.Tensor.size(x0)}')
        self.act_maps.append(x0)

        x05 = self.trans_conv05(x0)         #13x13
        # x05 = self.norm1(x05)
        x05 = F.gelu(x05)
        if self.verbose == 1: print(f'Activation map 0.5 dimmensions (conv) = {torch.Tensor.size(x05)}')
        self.act_maps.append(x05)

        x1 = self.trans_conv1(x05)        #13x13
        # x1 = self.norm1(x1)
        x1 = F.gelu(x1)
        if self.verbose == 1: print(f'Activation map 1 dimmensions (conv) = {torch.Tensor.size(x1)}')
        self.act_maps.append(x1)

        x2 = self.trans_conv2(x1)       #25x25
        # x2 = self.norm2(x2)
        x2 = F.gelu(x2)
        if self.verbose == 1: print(f'Activation map 2 dimmensions (conv) = {torch.Tensor.size(x2)}')
        self.act_maps.append(x2)

        x23 = self.trans_conv23(x2)       #25x25
        # x2 = self.norm2(x2)
        x23 = F.gelu(x23)
        if self.verbose == 1: print(f'Activation map 2 dimmensions (conv) = {torch.Tensor.size(x23)}')
        self.act_maps.append(x23)

        x27 = self.trans_conv27(x23)       #25x25
        # x2 = self.norm2(x2)
        x27 = F.gelu(x27 + x2)
        if self.verbose == 1: print(f'Activation map 2 dimmensions (conv) = {torch.Tensor.size(x27)}')
        self.act_maps.append(x27)

        x3 = self.trans_conv3(x27)       #50x50
        # x3 = self.norm3(x3)
        x3 = F.gelu(x3)
        if self.verbose == 1: print(f'Activation map 3 dimmensions (conv) = {torch.Tensor.size(x3)}')
        self.act_maps.append(x3)

        x4 = self.trans_conv4(x3)       #50x50
        # x4 = self.norm4(x4)
        x4 = F.gelu(x4 +x3)
        # x4 = F.gelu(x4)
        if self.verbose == 1: print(f'Activation map 4 dimmensions (conv) = {torch.Tensor.size(x4)}')
        self.act_maps.append(x4)

        out = self.trans_conv5(x4)      #100x100 # Without activation function.
        if self.verbose == 1: print(f'Output convolutional block dimmensions = {torch.Tensor.size(out)}')

        return out, self.act_maps

        
################################################