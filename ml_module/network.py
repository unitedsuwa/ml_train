import torch
import torch.nn as nn
import torch.nn.functional as F

class PlaneBlock(nn.Module):
    def __init__(self, inplane, outplane, stride=1, padding=1, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplane, outplane, kernel_size=3, stride=stride, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm2d(inplane)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.5)
        self.conv2 =  nn.Conv2d(outplane, outplane,  kernel_size=3, stride=1, padding=padding, bias=True)
        self.bn2 = nn.BatchNorm2d(outplane)

        # for change identity size
        if stride != 1 or inplane != outplane :
            self.downsample = nn.Sequential(
            nn.Conv2d(inplane, outplane, kernel_size=1, stride=stride, bias=True),
            nn.BatchNorm2d(outplane),
            )
        else :
            self.downsample = downsample

        print(inplane, outplane,)

    def forward(self, x):
        identity = x        # inputs shorcut

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        if self.downsample is not False :
            identity = self.downsample(x)
            #print(self.stride)

        out += identity
        # out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 128
        self.num_classes = 10

        # top convolution
        self.base = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1,),
                nn.Dropout2d(0.3),
        )

        # inner blocks settings
        layers =  [
                PlaneBlock(64, 64),
                PlaneBlock(64, 64),
                PlaneBlock(64, 64),
                PlaneBlock(64, 128, stride=2,),
                PlaneBlock(128, 128, stride=1,),
                PlaneBlock(128, 128, stride=1,),
                PlaneBlock(128, 128, stride=1,),
                PlaneBlock(128, 128, stride=1,),
                PlaneBlock(128, 256, stride=2,),
                PlaneBlock(256, 256, stride=1,),
                PlaneBlock(256, 256, stride=1,),
                PlaneBlock(256, 256, stride=1,),
                PlaneBlock(256, 256, stride=1,),
                PlaneBlock(256, 256, stride=1,),
                PlaneBlock(256, 512, stride=2,),
                PlaneBlock(512, 512, stride=1,),
                PlaneBlock(512, 512, stride=1,),
        ]

        self.planes = nn.Sequential(*layers)

        # pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # output side
        self.classifier = nn.Sequential(
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, self.num_classes),
        )


    def forward(self, x):
        x = self.base(x)
        x = self.planes(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x