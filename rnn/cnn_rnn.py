import torch.nn as nn
import torch.nn.functional as F
import torch

class Test_Classifier(nn.Module):
    def __init__(self):
        super(Test_Classifier, self).__init__()
        self.classifier = nn.Linear(64*64*64, 2)

    def forward(self, x):
        shape = x.shape
        x = x.reshape((shape[0], shape[1] , 64*64*64))
        x = self.classifier(x)
        return x

"""
CNN for feature extraction
"""
class simple_conv(nn.Module):
    def __init__(self,dropout=0.3, inter_num_ch = 16, img_dim = (64, 64, 64), feature_dim = 128):
        super(simple_conv, self).__init__()



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

        self.fc = nn.Linear(2 * inter_num_ch*(img_dim[0]//16)*(img_dim[1]//16)*(img_dim[2]//16), feature_dim) # output a 128 vector



    def forward(self, x):
        # create one more dim for the C, which is 1 for grey scale image
        x = torch.unsqueeze(x, 1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = torch.flatten(x, 1)

        x = self.fc(x) # output a 128 vector

        return x


"""
RNN for temporal processing
"""
class cnn_rnn(nn.Module):
    def __init__(self, cnn_dropout=0.3, inter_num_ch = 16, img_dim = (64, 64, 64), feature_size = 128, lstm_hidden_size = 16, rnn_dropout = 0.3, num_class = 2):
        super(cnn_rnn, self).__init__()

        self.feature_extractor = simple_conv(dropout=cnn_dropout, inter_num_ch = inter_num_ch, img_dim = img_dim, feature_dim = feature_size)

        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=lstm_hidden_size, num_layers=1,
                            batch_first=True, dropout=rnn_dropout)

        self.fc = nn.Linear(lstm_hidden_size, num_class)
        # self.fc = nn.Linear(128, num_class)


        # initialization for lstm
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 1.0)
            elif 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)


    def forward(self, x):

        # x shape: (N,T,64,64,64)
        N, T, H, W, D = x.shape

        # change shape to (N*T,64,64,64)
        x = x.view((N*T, H, W, D))

        # extract features:
        features = self.feature_extractor(x) # (N*T,128)

        features = features.view(N, T, -1) # (N, T, 128)

        output, _ = self.lstm(features) # (N, T, hidden_size)

        # convert to be num_classes:
        scores = self.fc(output) # (N, T, 2)

        return scores
