# -*- coding:utf-8 -*-
import jieba, MeCab, string, re
from configparser import ConfigParser

cfg = ConfigParser()
cfg.read("./config.cfg", encoding="utf8")

class Cls_To_Terms():
    """
    将字符转换为分词, 结果以列表的形式来返回
    """
    def __init__(self, email_body, lang):
        self.email_body = email_body
        self.lang = lang

    def to_terms_by_jieba(self):
        """
        利用 jieba 来进行分词
        :return: list
        """
        # 针对中文/英文进行分词
        # https://zhuanlan.zhihu.com/p/361052986
        # https://zhuanlan.zhihu.com/p/207057233
        jieba.load_userdict(r'./dict/zh_dict.txt')
        self.email_body = self.email_body.replace('\r', ' ').replace('\n', ' ')
        # 去掉特殊字符
        self.email_body = self.email_body.replace('。', ' ').replace('，', ' ').replace('（', ' ').replace('）', ' ')
        for i in string.punctuation:
            self.email_body = self.email_body.replace(i, ' ')
            self.email_body = self.email_body.lower()
        # 开始分词
        self.jieba_terms = jieba.lcut(self.email_body)
        # 去掉指定的项 https://www.cnblogs.com/sbj123456789/p/11252718.html
        n = 0
        for i in range(len(self.jieba_terms)):
            if self.jieba_terms[n] in [' ']:
                self.jieba_terms.pop(n)
            # 去掉所有的数字和英文
            elif re.match('[a-z]+|\d+', self.jieba_terms[n]):
                if self.lang != 'en':
                    self.jieba_terms.pop(n)
            else:
                n += 1
        # 调试部分
        if cfg.getboolean('Internal', 'Debug_Terms'):
            print("[Debug] {}".format(str(self.jieba_terms)))
        # 返回最终结果
        return self.jieba_terms

    def to_terms_by_mecab(self):
        """
        针对日语进行分词 https://github.com/SamuraiT/mecab-python3
        需要安装2个包, pip install mecab-python3 和 pip install unidic-lite
        :return: list
        """
        self.email_body = self.email_body.replace('\r', ' ').replace('\n', ' ')
        # 去掉特殊字符
        self.email_body = self.email_body.replace('。', ' ').replace('，', ' ').replace('（', ' ').replace('）', ' ')
        for i in string.punctuation:
            self.email_body = self.email_body.replace(i, ' ')
            self.email_body = self.email_body.lower()
        # 开始分词
        mecab_tagger = MeCab.Tagger("-Owakati")
        self.mecab_terms = mecab_tagger.parse(self.email_body).split()
        # 去掉指定的项
        n = 0
        for i in range(len(self.mecab_terms)):
            if self.mecab_terms[n] in [' ']:
                self.mecab_terms.pop(n)
            # 去掉所有的数字和英文
            elif re.match('[a-z]+|\d+', self.mecab_terms[n]):
                self.mecab_terms.pop(n)
            else:
                n += 1
        # 调试部分
        if cfg.getboolean('Internal', 'Debug_Terms'):
            print("[Debug] {}".format(str(self.mecab_terms)))
        # 返回最终结果
        return self.mecab_terms
