import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=100):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])

        self.fc_layer = nn.Sequential(
            # cifar10의 size가 32x32이므로
            nn.Linear(512*1*1, 4096),
            # 만약 imagenet대회 데이터인 224x224이라면
            # nn.Linear(512*7*7, 4096)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layer(out)

        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), 
                     nn.BatchNorm2d(x),   
                     # batchnormalization 층을 사용하는 모델들이 요즘 쓰임
                     # overfitting 억제와 학습속도 개선에 용이하다.
                     nn.ReLU(inplace=True)]
                in_channels = x
    

        return nn.Sequential(*layers)
    
    
    
def VGG11():
    return VGG(vgg_name = 'VGG11', num_classes = 100)

def VGG13():
    return VGG(vgg_name = 'VGG13', num_classes = 100)

def VGG16():
    return VGG(vgg_name = 'VGG16', num_classes = 100)

def VGG19():
    return VGG(vgg_name = 'VGG19', num_classes = 100)
