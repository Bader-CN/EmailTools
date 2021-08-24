# -*- coding:utf-8 -*-
import numpy as np
from configparser import ConfigParser

cfg = ConfigParser()
cfg.read(r"../config.cfg", encoding="utf8")

def fixTerms(text, fix_len=cfg.getint('Internal', 'Email_term_Length')):
    # 固定分词长度
    if len(text) > fix_len:
        text = text[:fix_len]
    else:
        text = text + (fix_len - len(text))*[0]
    return text

def terms_to_encoding(term_index, pd_data, cfg=cfg):
    """
    利用 term_index, 将分词进行编码, 并且将固定到统一的长度
    :param term_index: dict, 分词索引
    :param pd_data: Pandas.DataFrame
    :return: numpy, 编码后的分词数据
    """
    # x 为 pandas 中的每一行数据, term 为这一行中的每一个元素, 即分词
    # get(term, 0) 意味着如果获取到索引, 则使用索引代替, 否则就是用 0
    terms_data = pd_data.Body.apply(lambda x:[term_index.get(term, 0) for term in x])
    terms_data = terms_data.apply(fixTerms)
    # 将 Pandas 类型的数据转换为 Numpy 类型
    terms_data = np.array([line for line in terms_data])

    return terms_data
