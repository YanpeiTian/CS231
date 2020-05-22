import torch.nn as nn
import torch.nn.functional as F
import torch


class Test_Classifier(nn.Module):
    def __init__(self):
        super(Test_Classifier, self).__init__()
        self.classifier = nn.Linear(64*64*64, 2)

    def forward(self, x):
        x = x.reshape((-1, 64*64*64))
        x = self.classifier(x)
        return x

class simple_conv(nn.Module):
    def __init__(self,dropout=0.3, inter_num_ch = 16, img_dim = (64, 64, 64)):
        super(simple_conv, self).__init__()



        self.conv1 = nn.Sequential(
                        nn.Conv3d(1, inter_num_ch, kernel_size=3, padding=1),
                        nn.BatchNorm3d(inter_num_ch),
                        nn.ReLU(),
                        nn.MaxPool3d(2),
                        nn.Dropout3d(dropout))

        self.conv2 = nn.Sequential(
                        nn.Conv3d(inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1),
                        nn.BatchNorm3d(2 * inter_num_ch),
                        nn.ReLU(),
                        nn.MaxPool3d(2),
                        nn.Dropout3d(dropout))

        self.conv3 = nn.Sequential(
                        nn.Conv3d(2*inter_num_ch, 4*inter_num_ch, kernel_size=3, padding=1),
                        nn.BatchNorm3d(4 * inter_num_ch),
                        nn.ReLU(),
                        nn.MaxPool3d(2),
                        nn.Dropout3d(dropout))

        self.conv4 = nn.Sequential(
                        nn.Conv3d(4*inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1),
                        nn.BatchNorm3d(2 * inter_num_ch),
                        nn.ReLU(),
                        nn.MaxPool3d(2))

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1 = nn.Linear(2*inter_num_ch, 2)
        # self.fc2 = nn.Linear(inter_num_ch*(img_dim[0]//16)*(img_dim[1]//16)*(img_dim[2]//16), inter_num_ch*(img_dim[0]//16)*(img_dim[1]//16)*(img_dim[2]//16)//2)
        # self.fc3 = nn.Linear(inter_num_ch*(img_dim[0]//16)*(img_dim[1]//16)*(img_dim[2]//16)//2, 2)


    def forward(self, x):
        # create one more dim for the C, which is 1 for grey scale image
        x = torch.unsqueeze(x, 1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        # fc2 = self.fc2(fc1)
        # fc3 = self.fc3(fc2)
        return x
