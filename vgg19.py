from __future__ import print_function

import pdb
import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter


model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class SelectiveSequential(nn.Sequential):

    def __init__(self):
        super(SelectiveSequential, self).__init__()

    def forward(self, x, selected_layers):
        feature_maps = {}
        for name, module in self._modules.iteritems():
            x = module(x)
            if name in selected_layers:
                feature_maps[name] = x.clone()
        return feature_maps


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self._initialize_weights()

    def forward(self, x, selected_layers=None):
        return self.features(x, selected_layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def load_pretrained_model(self, pretrained_state_dict):

        # initialization
        custom_state_dict = self.state_dict()

        for name, value in zip(custom_state_dict.keys(), pretrained_state_dict.values()):

            try:
                custom_state_dict[name].copy_(value)
            except:
                print("skip loading key '{}' due to inconsistent size".format(name))

        self.load_state_dict(custom_state_dict)


def make_layers(cfg, batch_norm=False):
    layers = SelectiveSequential()
    in_channels = 3
    i, j = 1, 1
    for v in cfg:
        if v == 'M':
            name = 'maxpool' + str(i)
            layers.add_module(name, nn.MaxPool2d(kernel_size=2, stride=2))
            i, j = i+1, 1
        else:
            name = 'conv' + str(i) + '_' + str(j)
            layers.add_module(name, nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
            name = 'relu' + str(i) + '_' + str(j)
            layers.add_module(name, nn.ReLU(inplace=True))
            i, j = i, j+1
            in_channels = v
    return layers

cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_pretrained_model(model_zoo.load_url(model_urls['vgg19']))
    return model
