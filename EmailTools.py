# -*- coding:utf-8 -*-
import pandas as pd
from langdetect import detect
from configparser import ConfigParser
from win32com.client.gencache import EnsureDispatch as Dispatch

cfg = ConfigParser()
cfg.read("./config.cfg", encoding="utf8")

# 调整 Pandas 显示规则
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 3000)

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
            lang.append(detect(content))
        except Exception as e:
            pass
    lang = list(pd.value_counts(lang).index)
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
    :param src_mail:
    :param lang:
    :return:
    """

if __name__ == "__main__":
    src_mail = get_emails()
    # 判断获取原始邮件内容是否为空, 如果不为空, 则继续处理邮件内容
    if src_mail.Subject.count() != 0:
        src_mail, lang = chk_emails(src_mail)
        email_str = src_mail.Body[0].split('Subject:')[0]
        email_str = email_str.replace('\r','').replace('\n','')
        print(email_str[:200])
