# -*- coding:utf-8 -*-

import torch
import numpy as np
import pandas as pd
from mod.to_terms import zh_to_terms, en_to_terms, ja_to_terms
from sklearn.model_selection import train_test_split
from ml_lstm import RNN_LSTM_Net

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
# 生成分词索引
term_count = pd.value_counts(np.concatenate(pd_data.Body.values))
term_count = term_count[term_count > 2]
terms_list = list(term_count.index)
term_index = dict((term, terms_list.index(term) + 1) for term in terms_list)
# 利用分词索引, 将文本转换为数字编码
# x 为 pandas 中的每一行数据, term 为这一行中的每一个元素, 即分词
# get(term, 0) 意味着如果获取到索引, 则使用索引代替, 否则就是用 0
text_data = pd_data.Body.apply(lambda x:[term_index.get(term, 0) for term in x])
# 固定文本长度
def fixTerms(text, fix_len=16):
    if len(text) > fix_len:
        text = text[:fix_len]
    else:
        text = text + (fix_len - len(text))*[0]
    return text
text_data = text_data.apply(fixTerms)
# 将 Pandas 类型的数据转换为 Numpy 类型的数据
text_data = np.array([line for line in text_data])
# 处理标签
text_label = pd_data.Sentiment.values
# 拆分训练数据集和测试数据集
x_train, x_test, y_train, y_test = train_test_split(text_data, text_label, test_size=0.35)
# 创建 DataSet 数据集
class Mydataset(torch.utils.data.Dataset):
    def __init__(self, text_list, label_list):
        self.text_list = text_list
        self.label_list = label_list

    def __getitem__(self, index):
        text = torch.LongTensor(self.text_list[index])
        label = self.label_list[index]
        return text, label

    def __len__(self):
        return len(self.text_list)

train_ds = Mydataset(x_train, y_train)
test_ds = Mydataset(x_test, y_test)
# 创建 DataLoader 数据集
batch_size = 8
train_dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size = batch_size,
    shuffle = True,
)
test_dl = torch.utils.data.DataLoader(
    test_ds,
    batch_size=batch_size,
)
# 创建神经网络模型
max_terms=(len(term_count)+1)

model = RNN_LSTM_Net(max_terms)
loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 定义训练函数
def fit(epoch, model, trainloader, testloader):
    correct = 0
    total = 0
    running_loss = 0

    model.train()
    for x, y in trainloader:
        x = x.permute(1, 0)
        y_pred = model(x)
        loss = loss_fun(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()
    #    exp_lr_scheduler.step()
    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = correct / total

    test_correct = 0
    test_total = 0
    test_running_loss = 0

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

    print('epoch: ', epoch,
          'loss： ', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test_loss： ', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
          )

    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc

epochs = 3000
for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch, model, train_dl, test_dl)