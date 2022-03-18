"""
DentFCPN
============
Pasquale Lafiosca 2021
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
import utils
             
             
### Net Parts
class ConvInput(nn.Module):
    """
    2d input convolution 
    
    (B, in_channels, H, W) -> (B, out_channels, H, W)
    
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            PatchConv2d(in_channels, out_channels),  
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class Conv(nn.Module):
    """ 2d input convolution
    
    (B, in_channels, H, W) -> (B, out_channels, H, W)
    
    """
    def __init__(self, in_channels, out_channels, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=padding),  # 1=1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        return self.conv(x)


class ConvDown(nn.Module):
    """ 2d input convolution
    
    (B, in_channels, H, W) -> (B, out_channels, H/2, W/2)
    
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),  # /2
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)
        
        
class ConvUp(nn.Module):
    """
    Up-convolution
    
    (B, in_channels, H, W) -> (B, out_channels, 2*H, 2*W)
    """
    def __init__(self, in_channels, out_channels, padding=1):
        super().__init__()
        
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=padding, output_padding=1), # *2
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.upconv(x)
        

class Conv1x1(nn.Module):
    """
    Adapt number of features
    
    (B, in_channels, H, W) -> (B, out_channels, H, W)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0), # size out=in
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)
        

class ConvOut(nn.Module):
    """
    Same as Conv1x1 but without activation
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0), # size out=in
            nn.BatchNorm2d(out_channels),
            #nn.Sigmoid(), # Not needed if using BCEWithLogitsLoss
        )
    
    def forward(self, x):
        return self.conv(x)
         
               
  
### Model
class DentNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # LEVEL 0 (INPUT)
        self.encoder_convIn0 = Conv(config.NUM_INPUT_FEATURES, 64)
        self.skip0 = Conv1x1(64, 32)
        
        # LEVEL 1 (going down)
        self.encoder_downConv1A = ConvDown(64, 128) # (H/2, W/2)
        self.encoder_conv1B = Conv(128, 128)
        self.skip1 = Conv1x1(128, 64)              # Make "relations" among channels
        
        # LEVEL 2  (going down)
        self.encoder_downConv2A = ConvDown(128, 256) # (H/4, W/4)
        self.encoder_conv2B = Conv(256, 256)
        self.skip2 = Conv1x1(256, 128)  
        
        # LEVEL 3  (going down)
        self.encoder_downConv3A = ConvDown(256, 512) # (H/8, W/8)
        self.encoder_conv3B = Conv(512, 512)
        self.skip3 = Conv1x1(512, 256)  
        
        # LEVEL 4  (deepest)
        self.encoder_downConv4A = ConvDown(512, 1024) # (H/16, W/16)
        self.encoder_conv4B = Conv(1024, 1024)
        
        # LEVEL 3 (going up)
        self.encoder_upConv3A = ConvUp(1024,512)      # (H/8, W/8)
        # Add skip3
        self.decoder_conv3B = Conv(512+256, 512) 
        
        # LEVEL 2 (going up)
        self.encoder_upConv2A = ConvUp(512,256)        # (H/4, W/4)
        # Add skip2
        self.decoder_conv2B = Conv(256+128, 256) 
        
        # LEVEL 1 (going up)
        self.decoder_upconv1A = ConvUp(256, 128)       # (H/2, W/2)
        # Add skip1
        self.decoder_conv1B = Conv(128+64, 128)
        
        # LEVEL 0 (OUTPUT)
        self.decoder_upConvOut0A = ConvUp(128, 128)     # (H, W)
        # Add skip0
        self.decoder_convOut0B = Conv(128+32, 64)
        self.decoder_convOut0C = ConvOut(64, config.NUM_OUTPUT_FEATURES)
        
    def forward(self, features):
        """
        Parameters
        ----------
        features : torch.Tensor
            Features with shape (batch_size, C, H, W) where C is
            the number of channels. 
            
            
        """
        #batch_size, C, H, W = features.shape
            
        # LEVEL 0 (INPUT)
        x = self.encoder_convIn0(features)    # (H, W)
        skip0 = self.skip0(x)              
        
        # LEVEL 1 (going down)
        x = self.encoder_downConv1A(x)
        x = self.encoder_conv1B(x)
        skip1 = self.skip1(x)               # (H/2, W/2)
        
        # LEVEL 2 (going down)
        x = self.encoder_downConv2A(x)      # (H/4, W/4)
        x = self.encoder_conv2B(x)
        skip2 = self.skip2(x)               
        
        # LEVEL 3 (going down)
        x = self.encoder_downConv3A(x)      # (H/8, W/8)
        x = self.encoder_conv3B(x)
        skip3 = self.skip3(x)               
        
        # LEVEL 4 (deepest)
        x = self.encoder_downConv4A(x)      # (H/16, W/16)
        x = self.encoder_conv4B(x)
        
        # LEVEL 3 (going up)
        x = self.encoder_upConv3A(x)      # (H/8, W/8)
        # Add skip3
        x = torch.cat((x, skip3), dim=1)
        x = self.decoder_conv3B(x)
        
        # LEVEL 2 (going up)
        x = self.encoder_upConv2A(x)      # (H/4, W/4)
        # Add skip2
        x = torch.cat((x, skip2), dim=1)
        x = self.decoder_conv2B(x)
        
        # LEVEL 1 (going up)
        x = self.decoder_upconv1A(x)      # (H/2, W/2)
        # Add skip1
        x = torch.cat((x, skip1), dim=1)
        x = self.decoder_conv1B(x)
        
        # LEVEL 0 (OUTPUT)
        x = self.decoder_upConvOut0A(x)      # (H, W)
        
        # Add skip0
        x = torch.cat((x, skip0), dim=1)
        x = self.decoder_convOut0B(x)
        x = self.decoder_convOut0C(x)
        
        return x

        
if __name__ == "__main__":
    # Get number of total parameters
    model = DentNet()
    params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
    print("Total trainable parameters:", params)    
         
        






