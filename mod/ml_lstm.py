# -*- coding:utf-8 -*-

import torch
from configparser import ConfigParser

cfg = ConfigParser()
cfg.read(r"../config.cfg", encoding="utf8")

# 定义 RNN LSTM 网络模型
class RNN_LSTM_Net(torch.nn.Module):
    def __init__(self, max_terms):
        super(RNN_LSTM_Net, self).__init__()
        # 读取配置文件中的变量
        self.cfg = cfg
        self.embeding_dim = self.cfg.getint('Internal', 'lstm_embeding_dim')
        self.hidden_size = self.cfg.getint('Internal', 'lstm_hidden_size')
        self.num_layers = self.cfg.getint('Internal', 'lstm_num_layers')
        self.lstm_dropout = self.cfg.getfloat('Internal', 'lstm_dropout')
        self.dropout = self.cfg.getfloat('Internal', 'lstm_dropout')
        self.fc_dropout = self.cfg.getfloat('Internal', 'lstm_fc_dropout')
        self.fc_numbers = self.cfg.getint('Internal', 'lstm_fc_numbers')
        # 定义神经网络的各个层的特征
        self.ebdg = torch.nn.Embedding(max_terms, self.embeding_dim)
        self.lstm = torch.nn.LSTM(self.embeding_dim, self.hidden_size, num_layers=self.num_layers, dropout=self.lstm_dropout, bidirectional=True)
        self.fc01 = torch.nn.Linear(self.hidden_size*2, self.fc_numbers)
        self.fc02 = torch.nn.Linear(self.fc_numbers, self.fc_numbers)
        self.fc03 = torch.nn.Linear(self.fc_numbers, 2)
        self.drop = torch.nn.Dropout(self.fc_dropout)
    # 定义前向运算
    def forward(self, inputs):
        # out 是经过 embeding 后的结果
        out = self.ebdg(inputs)
        # lstm 会产生2个输出, 一个是 output, 另一个是 h_t, 其中 output[-1] 就是最后的 h_t
        output, _ = self.lstm(out)
        output = output[-1]
        fcdata = torch.relu(self.fc01(output))
        fcdata = self.drop(fcdata)
        fcdata = torch.sigmoid(self.fc02(fcdata))
        fcdata = self.drop(fcdata)
        fcdata = self.fc03(fcdata)
        return fcdata