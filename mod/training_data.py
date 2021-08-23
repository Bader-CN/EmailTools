# -*- coding:utf-8 -*-
import torch
from sklearn.model_selection import train_test_split

class to_dataset(torch.utils.data.Dataset):
    def __init__(self, text_list, label_list):
        self.text_list = text_list
        self.label_list = label_list

    def __getitem__(self, index):
        text = torch.LongTensor(self.text_list[index])
        label = self.label_list[index]
        return text, label

    def __len__(self):
        return len(self.text_list)

def to_dataloader(terms_data, text_label, batch_size=8, test_size=0.2):
    """
    将相关数据整理成 dataloader, 并且拆分成训练数据和测试数据
    :return: train_dl, test_dl
    """
    # 拆分训练数据集和测试数据集
    x_train, x_test, y_train, y_test = train_test_split(terms_data, text_label, test_size=test_size)
    # 生成对应的 dataset
    test_ds = to_dataset(x_test, y_test)
    train_ds = to_dataset(x_train, y_train)
    # 生成对应的 dataloader
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
    )

    return train_dl, test_dl