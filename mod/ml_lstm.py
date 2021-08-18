# -*- coding:utf-8 -*-

import torch
import numpy as np
import pandas as pd
from mod.to_terms import zh_to_terms, en_to_terms, ja_to_terms

# 预处理训练数据
pd_src_data = pd.read_csv(r'../data/email_data.csv', encoding='utf8')
# 处理每一种语言的分词数据
zh_train_data = pd_src_data[pd_src_data.Language.str.contains('zh')]
zh_train_data.reset_index(drop=True, inplace=True)
zh_train_data.Body = zh_train_data.Body.apply(zh_to_terms)
en_train_data = pd_src_data[pd_src_data.Language.str.contains('en')]
en_train_data.reset_index(drop=True, inplace=True)
en_train_data.Body = en_train_data.Body.apply(en_to_terms)
ja_train_data = pd_src_data[pd_src_data.Language.str.contains('ja')]
ja_train_data.reset_index(drop=True, inplace=True)
ja_train_data.Body = ja_train_data.Body.apply(ja_to_terms)
# 将这三类数据进行合并
pd_terms_data = pd.concat([zh_train_data, en_train_data, ja_train_data])
pd_terms_data.reset_index(drop=True, inplace=True)

# 生成分词索引
word_count = pd.value_counts(np.concatenate(pd_terms_data.Body.values))
# word_count = word_count[word_count > 2]
word_list = list(word_count.index)
word_index = dict((word, word_list.index(word) + 1) for word in word_list)
print(word_index)
