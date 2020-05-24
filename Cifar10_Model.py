# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:29:04 2020

@author: H343380
"""


import torch.nn as nn
import torch.nn.functional as F


class Cifar10(nn.Module):
    def __init__(self):
        super(Cifar10, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=1, stride=1), 
            nn.BatchNorm2d(32),
            nn.Dropout(0.25),
            nn.ReLU()
        )
        #Output=32*32 RF=3X3 [RFin + (Ksize-1 * JMPin) => 1+(3-1)*1 =3]  :JMPin=1, Jout= JMPin X s = 1
        
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1,groups=32) 
        )
        #Output=32*32 RF=5X5  [RFin + (Ksize-1 * JMPin) => 3+(3-1)*1 =5] :JMPin=1, Jout =JMPin X s =1
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.25),
            nn.ReLU()    
        )

        #Output=32*32 RF=5X5  [RFin + (Ksize-1 * JMPin) => 5+(1-1)*1 =5] :JMPin=1, Jout =JMPin X s =1

        self.pool1 = nn.MaxPool2d(2, 2)

        #Output=16*16 RF=6X6 [RFin + (Ksize-1 * JMPin) => 5+(2-1)*1 =6] :JMPin=1, Jout=  JMPin X s =2

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
            nn.ReLU()
        )
        #Output=16*16 RF=10X10 [RFin + (Ksize-1 * JMPin) => 6+(3-1)*2 =10] : Jout= JMPin X s = 2X1 :JMPin=2, Jout= JMPin X s = 2X1=2

        self.dilated = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1, stride=1,dilation=2),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
            nn.ReLU()
        )
        #Output=14*14 RF=14*14 [RFin + (Ksize-1 * JMPin) => 10+(3-1)*2 =14] : Jout= JMPin X s = 2X1 :JMPin=2, Jout= JMPin X s = 2X1=2

        self.pool2 = nn.MaxPool2d(2, 2)
        
        #Output=7*7 RF=16*16 [RFin + (Ksize-1 * JMPin) => 14+(2-1)*2 =16] : Jout= JMPin X s = 2X1 :JMPin=2, Jout= JMPin X s = 2X2=4

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.Dropout(0.25),
            nn.ReLU()
        )
        #Output=7*7 RF=24*24 [RFin + (Ksize-1 * JMPin) => 16+(3-1)*4 =24] : Jout= JMPin X s = 4*1=4

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.Dropout(0.25),
            nn.ReLU()
        )
        #Output=7*7 RF=32*32 [RFin + (Ksize-1 * JMPin) => 24+(3-1)*4 =32] :  Jout= JMPin X s = 4*1=4

        self.pool3 = nn.MaxPool2d(2, 2)
        
        #Output=3*3 RF=36X36 [RFin + (Ksize-1 * JMPin) => 32+(2-1)*4 =36] :  Jout= JMPin X s = 4X2=8

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.Dropout(0.25),
            nn.ReLU()
        )
        #Output=3*3 RF=52X52 [RFin + (Ksize-1 * JMPin) => 36+(3-1)*8 =52] : Jout= JMPin X s = 8*1=8

        self.gap =nn.AvgPool2d(kernel_size=2) ## 1*1*128
        self.fc1 = nn.Linear(1 * 1 * 128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
      x = self.conv1(x)
      x = self.depthwise(x)
      x = self.pointwise(x)
      x = self.pool1(x)
      x = self.conv3(x)
      x = self.dilated(x)
      x = self.pool2(x)
      x = self.conv5(x)
      x = self.conv6(x)
      x = self.pool3(x)
      x = self.conv7(x)  
      x = self.gap(x)
      x = x.view(-1, 128 * 1 * 1)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      x = x.view(-1, 10)
      return F.log_softmax(x)