import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1,downsample=None):
        expansion = 1
        
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        

        self.conv1=nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(outchannel)
        self.re1=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2=nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.re1(out)
        out=self.conv2(out)
        out=self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
     
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        yy=torch.zeros(1,512,4,4)
        self.inchannel = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bt =  nn.BatchNorm2d(64)
        self.r =  nn.ReLU()
            
        self.convqq=nn.Conv2d(512,64,kernel_size=1,stride=1,padding=0,bias=False)
        self.layer1 = self.make_layer(ResidualBlock, 64,  1, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 1, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 1, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 1, stride=2)
        self.fc = nn.Linear(512, num_classes)


    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    def _upsample(self,x,y):
        _,_,H,W = x.size()
        return F.upsample(y, size=(H,W), mode='bilinear') 

    def forward(self, x, y):
        out = self.conv1(x)
        print('1',out.shape)#([1, 64, 32, 32])
        #y=(1, 512, 4, 4)
        yy=self.convqq(y)#y=(1, 64, 4, 4)
        qq=self._upsample(out,yy)#qq=(1, 64, 32, 32)
        ooo=qq+out
        #qq+out=ooo
        out = self.layer1(ooo)
        print('l1',out.shape)#([1, 64, 32, 32])
        out = self.layer2(out)
        print('l2',out.shape)#([1, 128, 16, 16])
        out = self.layer3(out)
        print('l3',out.shape)#[1, 256, 8, 8])
        out4 = self.layer4(out)
        # print('l4',out4.shape)#[1, 512, 4, 4])
        out = F.avg_pool2d(out4, 4)
        # print('6',out.shape)#（1，512，4，4）
        out = out.view(out.size(0), -1)
        # print('7',out.shape)
        out = self.fc(out)
        # print(out.shape)
        return out,out4


# def ResNet18():
#     model = ResNet(ResidualBlock),
#     c1=model.conv1
#     print(c1)

    


def ResNet18():
    
    return ResNet(ResidualBlock)

# input=torch.randn(1,3,32,32)
# convqq=nn.Conv2d(512,64,kernel_size=1,stride=1,padding=0,bias=False)
# x=torch.randn(1,64,32,32)
# y=torch.randn(1,512,4,4) 
# qq=convqq(y)
# print(convqq(y).shape)#[1, 64, 4, 4])
# q=upsample(x,qq)
# print(q.shape)#[1, 64, 32, 32])

# model=ResNet18()
# outputs=model(input)
# qqqq=outputs[1]
# print(outputs[1].shape)

# for name,param in model.named_parameters():
#        print(name,param.shape)
