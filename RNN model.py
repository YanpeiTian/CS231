import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F
import numpy as np

class MultipleTimestepLSTM(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(64,64,64), z_dim=512,inter_num_ch=16, fc_num_ch=16, lstm_num_ch=16, kernel_size=3, name ='LSTM',
                conv_act='relu', requires_grad=True,fc_act='tanh', num_cls=2, num_timestep=5, skip_missing=True, init_lstm=False, rnn_type='GRU', fe_arch='Zucks', vae =None):
        super(MultipleTimestepLSTM, self).__init__()
        self.name =name
        self.z_dim=z_dim
        if fe_arch == 'baseline':
            self.feature_extractor = FeatureExtractor(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act)
            num_feat = int(inter_num_ch * (img_size[0]*img_size[1]*img_size[2]) / ((2**4)**3))
        elif fe_arch == 'resnet' or fe_arch == 'resnet_small':
            self.feature_extractor = FeatureExtractor_ResNet(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act, arch=fe_arch)
            if fe_arch == 'resnet':
                num_feat = 4*inter_num_ch*9
            elif fe_arch == 'resnet_small':
                num_feat = 36*4
        elif fe_arch == 'ehsan':
            self.feature_extractor = FeatureExtractor_Ehsan(in_num_ch, img_size, inter_num_ch, kernel_size, conv_act)
            num_feat = 4*inter_num_ch * 8
        elif fe_arch == 'Zucks':
            self.feature_extractor = vae
            num_feat = 512
        if fc_act == 'tanh':
            fc_act_layer = nn.Tanh()
        elif fc_act == 'relu':
            fc_act_layer = nn.ReLU()
        else:
            raise ValueError('No implementation of ', fc_act)
        if requires_grad== False:
            for p in self.parameters():
                p.requires_grad = False
        if num_cls == 2 or num_cls == 0:
            num_output = 1
        else:
            num_output = num_cls
        self.num_cls = num_cls
        self.dropout_rate = 0.1
        self.skip_missing = skip_missing
        self.fc1 = nn.Sequential(
                        nn.Linear(num_feat, fc_num_ch),
                        fc_act_layer)
                        # ,
                        # nn.Dropout(self.dropout_rate))

        # self.fc1 = nn.Sequential(
        #                 nn.Linear(num_feat, 4*fc_num_ch),
        #                 fc_act_layer,
        #                 nn.Dropout(self.dropout_rate))

        # self.fc2 = nn.Sequential(
        #                 nn.Linear(4*fc_num_ch, fc_num_ch),
        #                 fc_act_layer),
        #                 nn.Dropout(self.dropout_rate))


        if rnn_type == 'LSTM':
            self.lstm = nn.LSTM(input_size=fc_num_ch, hidden_size=lstm_num_ch, num_layers=1,
                            batch_first=True)
        elif rnn_type == 'GRU':
            self.lstm = nn.GRU(input_size=fc_num_ch, hidden_size=lstm_num_ch, num_layers=1,
                            batch_first=True)
        else:
            raise ValueError('No RNN Layer!')

        self.fc3 = nn.Linear(lstm_num_ch, num_output)

        if init_lstm:
            self.init_lstm()

    def init_lstm(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 1.0)
            elif 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, mask):
        #pdb.set_trace()
        bs, ts = x.shape[0], x.shape[1]
        x = torch.cat([x[b,...] for b in range(bs)], dim=0)  # (bs,ts,32,64,64) -> (bs*ts,32,64,64)
        x = x.unsqueeze(1)  # (bs*ts,1,32,64,64)
        out_z = self.feature_extractor.enc.encode_nkl(x)   # (bs*ts,512)
        # conv4_flatten = conv4.view(conv4.shape[0], -1)
        fc1 = self.fc1(out_z)
        # fc2 = self.fc2(fc1) # (bs*ts,16)
        fc2_concat = fc1.view(bs, ts, -1)  # (bs, ts, 16)
        #pdb.set_trace()
        if self.skip_missing:
            num_ts_list = mask.sum(1)
            if (num_ts_list == 0).sum() > 0:
                pdb.set_trace()
            # TODO: if skipping middle ts, need change
            _, idx_sort = torch.sort(num_ts_list, dim=0, descending=True)
            _, idx_unsort = torch.sort(idx_sort, dim=0)
            num_ts_list_sorted = num_ts_list.index_select(0, idx_sort)
            fc2_concat_sorted = fc2_concat.index_select(0, idx_sort)
            fc2_packed = torch.nn.utils.rnn.pack_padded_sequence(fc2_concat_sorted, num_ts_list_sorted, batch_first=True)
            lstm_packed, _ = self.lstm(fc2_packed)
            lstm_sorted, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_packed, batch_first=True)
            lstm = lstm_sorted.index_select(0, idx_unsort)
            # print(lstm.shape)
        else:
            lstm, _ = self.lstm(fc2_concat) # lstm: (bs, ts, 16)
        if lstm.shape[1] != ts:
            pad = torch.zeros(bs, ts-lstm.shape[1], lstm.shape[-1])
            lstm = torch.cat([lstm, pad.cuda()], dim=1)
        #lstm_reshape = lstm.contiguous().view(ts*bs, -1)   # (ts*bs, 16)
        #fc3 = self.fc3(lstm_reshape)
        #output = fc3.view(bs, ts, -1)
        output = self.fc3(lstm)
        if self.num_cls == 0:
            output = F.relu(output)
        if self.skip_missing:
            tpm = [output[i, num_ts_list[i].long()-1, :].unsqueeze(0) for i in range(bs)]
            output_last = torch.cat(tpm, dim=0)
            return [output_last, output, fc2_concat]
        else:
            return [output[:,-1,:], output]



def compute_rnn_loss(output, mask, label):
    # output shape: (Batch_Size, ts, 1)
    # label shape: (Batch_Size, ts)
    # mask shape: (Batch)Size, ts)
    bce = torch.nn.BCEWithLogitsLoss(reduction='none')
    loss = bce(input=output[mask==1].squeeze(dim=1),target=label[mask==1])
    S = torch.nn.Sigmoid()
    pred = S(output[mask==1])
    y = label[mask==1]
    pred = [1 if pred[i] >=0.5 else 0 for i in range(len(pred))]
    print('total:',len(pred),'acc for batch:', sum([1 if pred[i]==y[i] else 0 for i in range(len(pred))])/len(pred))
    return loss.mean()

def evaluate_rnn_loss(output, mask, label):
    # output shape: (Batch_Size, ts, 1)
    # label shape: (Batch_Size, ts)
    # mask shape: (Batch)Size, ts)
    # bce = torch.nn.BCEWithLogitsLoss(reduction='none')
    # loss = bce(output[mask==1],label[mask==1])
    S = torch.nn.Sigmoid()
    pred = S(output[mask==1])
    y = label[mask==1]
    pred = [1 if pred[i] >=0.5 else 0 for i in range(len(pred))]
    res = [1 if pred[i]==y[i] else 0 for i in range(len(pred))]
    acc = sum(res)/len(pred)
    # print('total:',len(pred),'acc for batch:', sum([1 if pred[i]==y[i] else 0 for i in range(len(pred))])/len(pred))
    # return loss.mean()
    return acc, len(pred), pred, res 
