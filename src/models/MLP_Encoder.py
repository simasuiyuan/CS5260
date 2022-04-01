import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MlpEncoder(nn.Module):

    def __init__(self, input_size, output_size, hidden_size):
        super(MlpEncoder , self).__init__()
        self.layer1 = nn.Linear( input_size, hidden_size)
        self.layer2 = nn.Linear( hidden_size , output_size )
        
    def forward(self, x):
        y       = self.layer1(x)
        y_hat   = torch.relu(y)
        scores  = self.layer2(y_hat)
        return scores