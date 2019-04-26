import torch.nn as nn
import torchvision.models as models

from wildcat.pooling import WildcatPool2d, ClassWisePool
from wildcat.Attention_pooling import Attention


class ResNetAWSL(nn.Module):

    def __init__(self, model, num_classes, pooling=WildcatPool2d(), dense=False,kernel_size=1):
        super(ResNetAWSL, self).__init__()

        self.dense = dense

        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4)

        # classification layer
        num_features = model.layer4[1].conv1.in_channels
        self.classifier = nn.Sequential(
            nn.Conv2d(num_features, num_classes, kernel_size=kernel_size, stride=1, padding=0, bias=True))

        self.spatial_pooling = pooling

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        if not self.dense:
            x = self.spatial_pooling(x)
        return x

    def get_config_optim(self, lr, lrp):
        return [{'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.classifier.parameters()},
                {'params': self.spatial_pooling.parameters()}]


def resnet50_attention(num_classes,sizeMaps, pretrained=True,num_maps=1,kernel_size=1):
    model = models.resnet50(pretrained)
    pooling = nn.Sequential()
    #pooling.add_module('class_wise', ClassWisePool(num_maps))
    pooling.add_module('spatial', Attention(sizeMaps,num_maps,num_classes=num_classes))
    return ResNetAWSL(model, num_classes * num_maps, pooling=pooling,kernel_size=kernel_size)


def resnet101_attention(num_classes,sizeMaps,pretrained=True,num_maps=1,kernel_size=1):
    model = models.resnet101(pretrained)
    pooling = nn.Sequential()
    #pooling.add_module('class_wise', ClassWisePool(num_maps))
    pooling.add_module('spatial', Attention(sizeMaps,num_maps,num_classes=num_classes))
    return ResNetAWSL(model, num_classes * num_maps, pooling=pooling,kernel_size=kernel_size)
