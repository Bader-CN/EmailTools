# -*- coding:utf-8 -*-
import jieba, MeCab, string

class Cls_To_Terms():
    """
    将字符转换为分词, 结果以列表的形式来返回
    """
    def __init__(self, email_body, lang):
        self.email_body = email_body
        self.lang = lang
        if lang in ['zh-cn', 'zh-tw', 'en']:
            self.to_terms_by_jieba()
        else:
            self.to_terms_by_mecab()

    def to_terms_by_jieba(self):
        """
        利用 jieba 来进行分词
        :return: list
        """
        jieba.load_userdict(r'./dict/zh_dict.txt')
        self.email_body = self.email_body.replace('\r', ' ').replace('\n', ' ')
        # 去掉特殊字符
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
            else:
                n += 1
        # 返回最终结果
        print(self.jieba_terms)
        return self.jieba_terms

    def to_terms_by_mecab(self):
        self.email_body = self.email_body.replace('\r', ' ').replace('\n', ' ')
        # 去掉特殊字符
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
            else:
                n += 1
        # 返回最终结果
        print(self.mecab_terms)
        return self.mecab_terms
