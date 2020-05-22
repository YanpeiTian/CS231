import torch.nn as nn
import torch.nn.functional as F
import torch


class sequence_model(nn.Module):
    def __init__(self,dropout=0.3, inter_num_ch = 16, img_dim = (64, 64, 64)):
        super(sequence_model, self).__init__()

        self.finalwidth = 2 * inter_num_ch

        # self.conv1 = nn.Sequential(
        #                 nn.Conv3d(1, inter_num_ch, kernel_size=3, padding=1),
        #                 nn.BatchNorm3d(inter_num_ch),
        #                 nn.ReLU(),
        #                 nn.MaxPool3d(2),
        #                 nn.Dropout3d(dropout))
        #
        # self.conv2 = nn.Sequential(
        #                 nn.Conv3d(inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1),
        #                 nn.BatchNorm3d(2 * inter_num_ch),
        #                 nn.ReLU(),
        #                 nn.MaxPool3d(2),
        #                 nn.Dropout3d(dropout))
        #
        # self.conv3 = nn.Sequential(
        #                 nn.Conv3d(2*inter_num_ch, 4*inter_num_ch, kernel_size=3, padding=1),
        #                 nn.BatchNorm3d(4 * inter_num_ch),
        #                 nn.ReLU(),
        #                 nn.MaxPool3d(2),
        #                 nn.Dropout3d(dropout))
        #
        # self.conv4 = nn.Sequential(
        #                 nn.Conv3d(4*inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1),
        #                 nn.BatchNorm3d(2 * inter_num_ch),
        #                 nn.ReLU(),
        #                 nn.MaxPool3d(2))
        self.conv1 = nn.Sequential(
                        nn.Conv3d(1, inter_num_ch, kernel_size=3, padding=1, stride=2),
                        nn.BatchNorm3d(inter_num_ch),
                        nn.ReLU(),
                        nn.Dropout3d(dropout))

        self.conv2 = nn.Sequential(
                        nn.Conv3d(inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1, stride=2),
                        nn.BatchNorm3d(2 * inter_num_ch),
                        nn.ReLU(),
                        nn.Dropout3d(dropout))

        self.conv3 = nn.Sequential(
                        nn.Conv3d(2*inter_num_ch, 4*inter_num_ch, kernel_size=3, padding=1, stride=2),
                        nn.BatchNorm3d(4 * inter_num_ch),
                        nn.ReLU(),
                        nn.Dropout3d(dropout))

        self.conv4 = nn.Sequential(
                        nn.Conv3d(4*inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1, stride=2),
                        nn.BatchNorm3d(2 * inter_num_ch),
                        nn.ReLU())

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.lstm = nn.LSTM(input_size=2*inter_num_ch, hidden_size=2*inter_num_ch, batch_first=True)

        self.fc1 = nn.Linear(2*inter_num_ch, 2)

        self.test = nn.Linear(64*64*64, 2)


    def forward(self, x):
        # create one more dim for the C, which is 1 for grey scale image
        shape = x.shape
        x = x.reshape(-1,64,64,64)
        x = torch.unsqueeze(x, 1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        # x = x.reshape(shape[0], shape[1], -1)
        # x, _ = self.lstm(x)
        # x = x.reshape(-1, self.finalwidth)

        x = self.fc1(x)
        x = x.reshape(shape[0], shape[1], -1)

        # shape = x.shape
        # x = x.reshape(-1,64,64,64)
        # x = torch.flatten(x, 1)
        # x = self.test(x)
        # x = x.reshape(shape[0], shape[1], -1)

        return x
