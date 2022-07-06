from unet_model import UNet
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

def calc_iou(prediction, ground_truth):
    n_images = len(prediction)
    intersection, union = 0, 0
    for i in range(n_images):
        intersection += torch.logical_and(prediction[i] > 0, ground_truth[i] > 0).float().sum() 
        union += torch.logical_or(prediction[i] > 0, ground_truth[i] > 0).float().sum()
    return float(intersection + 1e-6) / (union + 1e-6)

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)       
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss       
        return Dice_BCE

backbone = UNet(n_channels=3, n_classes=64, bilinear=False)

class Decoder2Vector(torch.nn.Module):
    def __init__(self, num_out=1):
        super().__init__()
        self.pool = nn.AvgPool2d(256)
        self.fc = nn.Linear(64, num_out)   
    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
    def compute_loss(self, positive, negative):
        diff = negative - positive
        return torch.where(diff < 10, torch.log(torch.exp(diff) + 1), diff).mean()

class Decoder2Mat(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 1, 1)
    def forward(self, x):
        x = self.conv(x)
        x = torch.sigmoid(x)
        return x
    
class Image2VectorWithPairwise(torch.nn.Module):
    def __init__(self, num_out=1):
        super().__init__()
        self.encoder = backbone
        self.decoder = Decoder2Vector(num_out=1)    
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
    def compute_loss(self, positive, negative):
        diff = negative - positive
        return torch.where(diff < 10, torch.log(torch.exp(diff) + 1), diff).mean()
        
class Image2Image(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = backbone
        self.decoder = Decoder2Mat()
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
    def compute_loss(self, predictions, gt):
        loss = DiceBCELoss()
        return loss(predictions, gt)
    def post_processing(self, prediction):
        return prediction  > 0.5
    def metric(self, y_batch, y_pred):
        return calc_iou(y_pred, y_batch)

