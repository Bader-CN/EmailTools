# -*- coding:utf-8 -*-
import re, jieba, MeCab
import pandas as pd
from langdetect import detect
from configparser import ConfigParser
from win32com.client.gencache import EnsureDispatch as Dispatch
from mod.to_terms import Cls_To_Terms

cfg = ConfigParser()
cfg.read("./config.cfg", encoding="utf8")

# 调整 Pandas 显示规则
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 4000)

def get_emails():
    """
    根据指定条件, 获取邮件里面的相关信息
    :return:Pandas.DataFrame
    """
    # https://blog.csdn.net/weixin_43750377/article/details/108080662
    outlook = Dispatch("Outlook.Application")
    mapi = outlook.GetNamespace("MAPI")
    accounts = mapi.Folders
    # 这里仅仅返回最新的那一封邮件数据
    for email in accounts:
        if email.Name == cfg.get('Filters', 'By_Account_Name'):
            for folder in email.Folders:
                if folder.Name == cfg.get('Filters', 'By_Email_Folder'):
                    mails = folder.Items
                    mails.Sort("[ReceivedTime]", True)
                    for mail in mails:
                        pd_mail = pd.DataFrame([{"Subject":mail.Subject, "SenderName":mail.SenderName, "To":mail.To, "CC":mail.CC, "Body":mail.Body}])
                        break
                    # 过滤邮件标题
                    pd_mail = pd_mail[pd_mail.Subject.str.contains(cfg.get('Filters', 'By_Email_Subject'))]
                    # 返回 Pandas.DataFrame 数据
                    return pd_mail

def chk_emails(src_mail):
    """
    对传入的邮件进行预分析和检查, 并且给出是通过什么语言来写的
    :param src_mail: Pandas.DataFrame
    :return: Pandas.DataFrame, str(language)
    """
    # https://zhuanlan.zhihu.com/p/34246546
    lang = []
    body = src_mail.Body[0]
    # 判断邮件会话中最新的会话
    mail_ends = cfg.get('Email_Settings', 'Email_End_Str').split('|')
    for end in mail_ends:
        body = body.split(end, 1)[0]
    # 将内容逐行断开
    contents = body.split('\n')
    for content in contents:
        try:
            # 如果字符串里包含指定的字符, 则直接判定为该语言
            if re.match(cfg.get('Internal', 'ZH_TW'), content):
                lang.append('zh-tw')
            # 否则, 则进行自动语言判断
            else:
                lang.append(detect(content))
            # 调试使用
            if cfg.getboolean('Internal', 'Debug_Language'):
                print('[Debug] ' + detect(content) + "|||" + content)
        except Exception as e:
            pass
    lang = list(pd.value_counts(lang).index)
    # 调试部分, 用于显示语言类型
    if cfg.getboolean('Internal', 'Debug_Language'):
        print('[Debug] ' + str(lang))
    # 返回原始数据以及对应的语言类型
    if "zh-cn" in lang:
        return src_mail, "zh-cn"
    elif "zh-tw" in lang:
        return src_mail, "zh-tw"
    elif "ja" in lang:
        return src_mail, "ja"
    else:
        return src_mail, lang[0]

def pre_emails_to_terms(src_mail, lang):
    """
    根据邮件内容以及语言分类来进行预处理, 将邮件中的 Body 部分进行分词化
    :param src_mail: Pandas.DataFrame
    :param lang: str
    :return: Pandas.DataFrame
    """
    pre_mail = src_mail
    pre_mail.Body = pre_mail.Body.str.replace('\r|\n','', regex=True)
    pre_mail.Body = pre_mail.Body.apply(lambda x:x[:cfg.getint('Internal', 'Email_Body_Length')])
    # 针对中文/英文进行分词
    # https://zhuanlan.zhihu.com/p/361052986
    # https://zhuanlan.zhihu.com/p/207057233
    if lang in ['zh-cn', 'zh-tw', 'en']:
        # 加载自定义分词
        jieba.load_userdict(r'./dict/zh_dict.txt')
        pre_mail.Body = pre_mail.Body.str.replace('\,|\，|\.|\。|_|=|/|-|_|\+|\|', ' ', regex=True)
        pre_mail.Body = pre_mail.Body.apply(lambda x:jieba.lcut(x))
        zh_terms = pre_mail.Body[0]
        # 遍历一次数组, 去掉指定的项 https://www.cnblogs.com/sbj123456789/p/11252718.html
        n = 0
        for i in range(len(zh_terms)):
            if zh_terms[n] in [' ', '（', '）', '(', ')', '·', '<', '>', ':', '@', '-', '_']:
                zh_terms.pop(n)
            else:
                try:
                    # 尝试将大写转换为小写
                    zh_terms[n] = zh_terms[n].lower()
                except:
                    pass
                n += 1
        pre_mail.Body[0] == zh_terms

    # 针对日语进行分词 https://github.com/SamuraiT/mecab-python3
    # 需要安装2个包, pip install mecab-python3 和 pip install unidic-lite
    elif lang in ['ja']:
        mecab_tagger = MeCab.Tagger("-Owakati")
        pre_mail.Body = pre_mail.Body.str.replace('\,|\，|\.|\。|_|=|/|-|_|\+|\|', ' ', regex=True)
        pre_mail.Body = pre_mail.Body.apply(lambda x: (mecab_tagger.parse(x)).split())
        ja_terms = pre_mail.Body[0]
        n = 0
        for i in range(len(ja_terms)):
            if ja_terms[n] in [' ', '（', '）', '(', ')', '·', '<', '>', ':', '@', '-', '_']:
                ja_terms.pop(n)
            else:
                try:
                    # 尝试将大写转换为小写
                    ja_terms[n] = ja_terms[n].lower()
                except:
                    pass
                n += 1
        pre_mail.Body[0] == ja_terms

    # 调试部分, 用于显示分词信息
    if cfg.getboolean('Internal', 'Debug_Terms'):
        print('[Debug] ' + str(pre_mail.Body[0]))

    return pre_mail

if __name__ == "__main__":
    src_mail = get_emails()
    # 判断获取原始邮件内容是否为空, 如果不为空, 则继续处理邮件内容
    if src_mail.Subject.count() != 0:
        src_mail, lang = chk_emails(src_mail)
        # print(src_mail.Body[0])
        # pre_mail = pre_emails_to_terms(src_mail, lang)
        # print(pre_mail.SenderName)
        terms = Cls_To_Terms(src_mail.Body[0], lang)

