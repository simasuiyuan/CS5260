import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class EncoderCNN(nn.Module):
#     """Encoder inputs images and returns feature maps"""
#     def __init__(self):
#         super(EncoderCNN, self).__init__()
#         resnet = models.resnet152(pretrained=True)
#         for param in resnet.parameters():
#             param.requires_grad_(False)

#         modules = list(resnet.children())[:-2]
#         self.resnet = nn.Sequential(*modules)

#     def forward(self, images):
#         features = self.resnet(images)
#         # first, we need to resize the tensor to be
#         # (batch, size*size, feature_maps)
#         batch, feature_maps, size_1, size_2 = features.size()
#         features = features.permute(0, 2, 3, 1)
#         features = features.view(batch, size_1*size_2, feature_maps)

#         return features


class EncoderCNN(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        batch, feature_maps, size_1, size_2 = out.size()
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        out = out.view(batch, size_1*size_2, feature_maps) 
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune