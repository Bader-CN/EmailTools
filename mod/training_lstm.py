# -*- coding:utf-8 -*-

import torch, pickle
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mod.to_terms import zh_to_terms, en_to_terms, ja_to_terms
from ml_lstm import RNN_LSTM_Net
from mod.to_terms_idx import terms_to_index
from mod.to_terms_encoding import terms_to_encoding
from mod.training_data import to_dataloader

# 预处理训练数据
pd_data = pd.read_csv(r'../data/email_data.csv', encoding='utf8')
# 中文分词
pd_data_zh = pd_data[pd_data.Language.str.contains('zh')]
pd_data_zh.reset_index(drop=True, inplace=True)
pd_data_zh.Body = pd_data_zh.Body.apply(zh_to_terms)
# 英文分词
pd_data_en = pd_data[pd_data.Language.str.contains('en')]
pd_data_en.reset_index(drop=True, inplace=True)
pd_data_en.Body = pd_data_en.Body.apply(en_to_terms)
# 日语分词
pd_data_ja = pd_data[pd_data.Language.str.contains('ja')]
pd_data_ja.reset_index(drop=True, inplace=True)
pd_data_ja.Body = pd_data_ja.Body.apply(ja_to_terms)
# 合并所有的数据
pd_data = pd.concat([pd_data_zh, pd_data_en])
pd_data.reset_index(drop=True, inplace=True)
# 生成分词索引文件
terms_to_index(pd_data)
# 读取分词索引文件
with open('./term_index.pkl', 'rb') as file:
    term_index = pickle.load(file)
# 利用 term_index, 将分词进行编码
terms_data = terms_to_encoding(term_index, pd_data)
# 处理标签
text_label = pd_data.Sentiment.values
# 生成对应的 Dataloader
train_dl, test_dl = to_dataloader(terms_data, text_label, batch_size=4, test_size=0.4)
# 定义 LSTM 模型的相关参数
model = RNN_LSTM_Net(len(term_index)+1)
loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# 定义学习曲率衰减器
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
# 定义训练函数
def fit(epoch, model, trainloader, testloader):
    # 初始化变量(训练)
    correct = 0
    total = 0
    running_loss = 0
    # 训练模式
    print("目前的学习率：%f" % (optimizer.param_groups[0]['lr']))
    model.train()
    for x, y in trainloader:
        # 调换 x 的维度位置
        x = x.permute(1, 0)
        y_pred = model(x)
        loss = loss_fun(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        exp_lr_scheduler.step()
        with torch.no_grad():
            # 对 y_perd 取最大可能的值
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()
    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = correct / total

    # 初始化变量(测试)
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    # 测试模式
    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            x = x.permute(1, 0)
            y_pred = model(x)
            loss = loss_fun(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total

    print('迭代次数: ', epoch,
          '训练损失: ', round(epoch_loss, 6),
          '训练准度: ', round(epoch_acc, 6),
          '测试损失: ', round(epoch_test_loss, 6),
          '测试准度: ', round(epoch_test_acc, 6)
          )

    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc

# 开始训练并获取训练的相关数据
epochs = 300
acc_by_train = []
acc_by_test = []
loss_by_train = []
loss_by_test = []
for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch, model, train_dl, test_dl)
    acc_by_train.append(epoch_acc)
    acc_by_test.append(epoch_test_acc)
    loss_by_train.append(epoch_loss)
    loss_by_test.append(epoch_test_loss)

# 绘制训练图像
x = list(range(epochs))
plt.plot(x, acc_by_train, label="Training", c='r')
plt.plot(x, acc_by_test, label="Test", c='b')
plt.plot(x, loss_by_train, label="Loss Training")
plt.plot(x, loss_by_test, label="Loss Test")
plt.legend() # 显示图标
plt.title("LSTM Training Result")
plt.show()