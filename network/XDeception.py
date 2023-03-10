import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import copy


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=False)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class XDeception(nn.Module):

    def __init__(self):
        """ 
            Deception Network based on Xception
        """
        super(XDeception, self).__init__()

        # Sub-network1
        self.relu = nn.ReLU(inplace=False)
        model_list1 = [  nn.Conv2d(3, 16, 3, 2, 0, bias=False),
                         nn.BatchNorm2d(16),

                         nn.Conv2d(16, 32, 3, bias=False),
                         Block(32, 64, 2, 2, start_with_relu=False, grow_first=True),
                         Block(64, 128, 2, 2, start_with_relu=True, grow_first=True),
                         Block(128, 364, 2, 2, start_with_relu=True, grow_first=True),
                         Block(364, 364, 3, 1, start_with_relu=True, grow_first=True),
                         Block(364, 364, 3, 1, start_with_relu=True, grow_first=True),
                         Block(364, 364, 3, 1, start_with_relu=True, grow_first=True),
                         Block(364, 364, 3, 1, start_with_relu=True, grow_first=True),

                         Block(364, 364, 3, 1, start_with_relu=True, grow_first=True),
                         Block(364, 364, 3, 1, start_with_relu=True, grow_first=True),
                         Block(364, 364, 3, 1, start_with_relu=True, grow_first=True),
                         Block(364, 364, 3, 1, start_with_relu=True, grow_first=True),

                         Block(364, 512, 2, 2, start_with_relu=True, grow_first=False),

                         SeparableConv2d(512, 768, 3, 1, 1),
                         nn.BatchNorm2d(768),

                         SeparableConv2d(768, 1024, 3, 1, 1),
                         nn.BatchNorm2d(1024) ]
        
        # Sub-network2
        model_list2 = [  nn.Conv2d(3, 16, 3, 2, 0, bias=False),
                         nn.BatchNorm2d(16),

                         nn.Conv2d(16, 32, 3, bias=False),
                         Block(32, 64, 2, 2, start_with_relu=False, grow_first=True),
                         Block(64, 128, 2, 2, start_with_relu=True, grow_first=True),
                         Block(128, 364, 2, 2, start_with_relu=True, grow_first=True),
                         Block(364, 364, 3, 1, start_with_relu=True, grow_first=True),
                         Block(364, 364, 3, 1, start_with_relu=True, grow_first=True),
                         Block(364, 364, 3, 1, start_with_relu=True, grow_first=True),
                         Block(364, 364, 3, 1, start_with_relu=True, grow_first=True),

                         Block(364, 364, 3, 1, start_with_relu=True, grow_first=True),
                         Block(364, 364, 3, 1, start_with_relu=True, grow_first=True),
                         Block(364, 364, 3, 1, start_with_relu=True, grow_first=True),
                         Block(364, 364, 3, 1, start_with_relu=True, grow_first=True),

                         Block(364, 512, 2, 2, start_with_relu=True, grow_first=False),

                         SeparableConv2d(512, 768, 3, 1, 1),
                         nn.BatchNorm2d(768),

                         SeparableConv2d(768, 1024, 3, 1, 1),
                         nn.BatchNorm2d(1024) ]

        self.model1 = nn.ModuleList()
        self.model2 = nn.ModuleList()

        for layer in model_list1:
            self.model1.append(layer)
        
        for layer in model_list2:
            self.model2.append(layer)
        
        self.fc1 = nn.Linear(1024, 1)
        self.fc2 = nn.Linear(1024, 1)

    def features(self, input):
        relu_layer = [1, 4, 19]

        x1 = self.model1[0](input)
        x2 = self.model2[0](input)

        for i in range(1, len(self.model1)):
            x1 = self.model1[i](x1)
            x2 = self.model2[i](x2)
            if i in relu_layer:
                x1 = self.relu(x1)
                x2 = self.relu(x2)

        return x1, x2

    def logits(self, x1, x2):
        x1 = self.relu(x1)

        x1 = F.adaptive_avg_pool2d(x1, (1, 1))
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc1(x1)

        x2 = self.relu(x2)

        x2 = F.adaptive_avg_pool2d(x2, (1, 1))
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc2(x2)

        return x1, x2

    def forward(self, input):
        x1, x2 = self.features(input)
        x1, x2 = self.logits(x1, x2)

        return torch.cat((x1, x2), axis = 1)