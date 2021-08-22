# -*- coding:utf-8 -*-

import torch
from configparser import ConfigParser
# 尝试读取配置文件
cfg = ConfigParser()
try:
    cfg.read(r"./config.cfg", encoding="utf8")
except:
    cfg.read(r"../config.cfg", encoding="utf8")

# 定义 RNN LSTM 网络模型
class RNN_LSTM_Net(torch.nn.Module):
    def __init__(self, max_terms, embeding_dim, hidden_size):
        super(RNN_LSTM_Net, self).__init__()
        self.em = torch.nn.Embedding(max_terms, embeding_dim)  # 200*batch*100
        self.rnn = torch.nn.LSTM(embeding_dim, hidden_size, num_layers=2, dropout=0.5)  # batch*300
        self.fc1 = torch.nn.Linear(hidden_size, 1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.fc3 = torch.nn.Linear(1024, 2)
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, inputs):
        x = self.em(inputs)
        r_o, _ = self.rnn(x)
        r_o = r_o[-1]
        x = torch.relu(self.fc1(r_o))
        x = self.drop(x)
        x = torch.sigmoid(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x