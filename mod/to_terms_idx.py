# -*- coding:utf-8 -*-
import pickle
import numpy as np
import pandas as pd
from configparser import ConfigParser

cfg = ConfigParser()
cfg.read(r"../config.cfg", encoding="utf8")

def terms_to_index(terms):
    """
    根据分词的信息来返回每个分词所对应的索引
    :param terms:
    :return:
    """
    # 生成分词索引
    term_count = pd.value_counts(np.concatenate(terms.Body.values))
    term_count = term_count[term_count > cfg.getint('Internal', 'min_term_count')]
    terms_list = list(term_count.index)
    term_index = dict((term, terms_list.index(term) + 1) for term in terms_list)
    # 将分词索引保存起来
    with open('./term_index.pkl', 'wb') as file:
        pickle.dump(term_index, file)

    # # 利用分词索引, 将文本转换为数字编码
    # # x 为 pandas 中的每一行数据, term 为这一行中的每一个元素, 即分词
    # # get(term, 0) 意味着如果获取到索引, 则使用索引代替, 否则就是用 0
    # text_data = pd_data.Body.apply(lambda x: [term_index.get(term, 0) for term in x])
    #
    # # 固定文本长度
    # def fixTerms(text, fix_len=16):
    #     if len(text) > fix_len:
    #         text = text[:fix_len]
    #     else:
    #         text = text + (fix_len - len(text)) * [0]
    #     return text
    #
    # text_data = text_data.apply(fixTerms)
    # # 将 Pandas 类型的数据转换为 Numpy 类型的数据
    # text_data = np.array([line for line in text_data])