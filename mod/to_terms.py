# -*- coding:utf-8 -*-
import jieba, MeCab, string, re, os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def zh_to_terms(mail_content):
    """
    针对 zh-cn 和 zh-tw 进行分词
    https://zhuanlan.zhihu.com/p/361052986
    https://zhuanlan.zhihu.com/p/207057233
    :return: list
    """
    jieba.load_userdict(os.path.join(base_dir, 'dict', 'zh_dict.txt'))
    mail_content = mail_content.replace('\r', ' ').replace('\n', ' ')
    # 去掉特殊字符
    mail_content = mail_content.replace('。', ' ').replace('，', ' ').replace('（', ' ').replace('）', ' ').replace('', ' ').replace('\t', ' ').replace('※', ' ').replace('＞', ' ')
    for i in string.punctuation:
        mail_content = mail_content.replace(i, ' ')
        mail_content = mail_content.lower()
    # 开始分词
    zh_jieba_terms = jieba.lcut(mail_content)
    # 去掉指定的项 https://www.cnblogs.com/sbj123456789/p/11252718.html
    n = 0
    for i in range(len(zh_jieba_terms)):
        if zh_jieba_terms[n] in [' ']:
            zh_jieba_terms.pop(n)
        # 去掉所有的数字和英文
        elif re.match('[a-z]+|\d+', zh_jieba_terms[n]):
            zh_jieba_terms.pop(n)
        else:
            n += 1
    # 返回最终结果
    return zh_jieba_terms

def en_to_terms(mail_content):
    """
    针对 en 进行分词
    https://zhuanlan.zhihu.com/p/361052986
    https://zhuanlan.zhihu.com/p/207057233
    :return: list
    """
    mail_content = mail_content.replace('\r', ' ').replace('\n', ' ')
    # 去掉特殊字符
    mail_content = mail_content.replace('。', ' ').replace('，', ' ').replace('（', ' ').replace('）', ' ').replace('', ' ').replace('\t', ' ').replace('※', ' ').replace('＞', ' ')
    for i in string.punctuation:
        mail_content = mail_content.replace(i, ' ')
        mail_content = mail_content.lower()
    # 开始分词
    en_jieba_terms = jieba.lcut(mail_content)
    # 去掉指定的项 https://www.cnblogs.com/sbj123456789/p/11252718.html
    n = 0
    for i in range(len(en_jieba_terms)):
        if en_jieba_terms[n] in [' ']:
            en_jieba_terms.pop(n)
        else:
            n += 1
    # 返回最终结果
    return en_jieba_terms

def ja_to_terms(mail_content):
    """
    针对日语进行分词 https://github.com/SamuraiT/mecab-python3
    需要安装2个包, pip install mecab-python3 和 pip install unidic-lite
    :return: list
    """
    mail_content = mail_content.replace('\r', ' ').replace('\n', ' ')
    # 去掉特殊字符
    mail_content = mail_content.replace('。', ' ').replace('，', ' ').replace('（', ' ').replace('）', ' ')
    for i in string.punctuation:
        mail_content = mail_content.replace(i, ' ')
        mail_content = mail_content.lower()
    # 开始分词
    mecab_tagger = MeCab.Tagger("-Owakati")
    mecab_terms = mecab_tagger.parse(mail_content).split()
    # 去掉指定的项
    n = 0
    for i in range(len(mecab_terms)):
        if mecab_terms[n] in [' ']:
            mecab_terms.pop(n)
        # 去掉所有的数字和英文
        elif re.match('[a-z]+|\d+', mecab_terms[n]):
            mecab_terms.pop(n)
        else:
            n += 1

    # 返回最终结果
    return mecab_terms