import torch as t
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchsummary import summary


class Localization_anchor_net(nn.Module):
    def __init__(self, class_num=5, pretrained=True, anchor_num_per_point=3,
                 dropout_or_not=False,prob=0.5):  # without the background
        self.class_num = class_num
        self.anchor_num_per_point = anchor_num_per_point

        super(Localization_anchor_net, self).__init__()
        resnet18 = models.resnet18(pretrained=pretrained)
        self.feature_extraction = nn.Sequential()
        for name, each in resnet18.named_children():
            if (not isinstance(each, nn.AdaptiveAvgPool2d)) and (not isinstance(each, nn.Linear)):
                self.feature_extraction.add_module(name=name, module=each)

        self.dropout_flag = dropout_or_not
        self.dropout = nn.Dropout2d(prob)

        self.sliding_window = nn.Conv2d(512,512,3,1,1,bias = False)
        self.bn_sw = nn.BatchNorm2d(512)

        self.regre_head = nn.Conv2d(512,self.anchor_num_per_point * 4,3,1,1,bias = True)
        self.class_head = nn.Conv2d(512,self.anchor_num_per_point*(self.class_num+1),3,1,1,bias = True)

    def forward(self, x):
        self.features = self.feature_extraction(x)
        _,_,hh,ww = self.features.shape

        self.features = F.relu(self.bn_sw(self.sliding_window(self.features)))

        if self.dropout_flag:
            self.features = self.dropout(self.features)

        self.regre = self.regre_head(self.features)
        self.regre = self.regre.permute(0,2,3,1).contiguous()
        self.regre = self.regre.view(self.regre.shape[0],
            self.regre.shape[1]*self.regre.shape[2]*self.anchor_num_per_point,4)

        self.classification = self.class_head(self.features)
        self.classification = self.classification.permute(0,2,3,1).contiguous()
        self.classification = self.classification.view(self.classification.shape[0],
            self.anchor_num_per_point*self.classification.shape[1]*self.classification.shape[2],
                                                       self.class_num+1)

        return self.regre,self.classification


if __name__ == '__main__':
    import time

    input = t.zeros((1, 3, 128, 128)).cpu()
    net = Localization_anchor_net(class_num=5, pretrained=True, anchor_num_per_point=3).cpu()

    start = time.time()
    regre,classification = net(input)
    end = time.time()
    print(regre.size())
    print(classification.size())
    print(end - start)

    print(summary(net, input_size=(3, 128, 128), device='cpu'))