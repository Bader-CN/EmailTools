# -*- coding:utf-8 -*-
import pickle, os
import numpy as np
import pandas as pd
from configparser import ConfigParser

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = ConfigParser()
cfg.read(os.path.join(base_dir, 'config.cfg'), encoding="utf8")

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
    with open(os.path.join(base_dir, 'train', 'term_index.pkl'), 'wb') as file:
        pickle.dump(term_index, file)
    # 生成文本文件
    with open(os.path.join(base_dir, 'train', 'term_index.txt'), 'w', encoding='utf8') as file:
        file.write(str(term_index))