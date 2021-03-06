# -*- coding:utf-8 -*-
import os, re, pickle
import torch
import numpy as np
import pandas as pd
import win32api,win32con
from langdetect import detect
from configparser import ConfigParser
from win32com.client.gencache import EnsureDispatch as Dispatch
from mod.to_terms import zh_to_terms, en_to_terms, ja_to_terms
from mod.to_terms_encoding import terms_to_encoding
from mod.ml_lstm import RNN_LSTM_Net

cfg = ConfigParser()
base_dir = os.path.dirname(os.path.abspath(__file__))
cfg_file = os.path.join(base_dir, 'config.cfg')
cfg.read(cfg_file, encoding="utf8")

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
    判断邮件的最新内容, 并且将最新的邮件内容处理为分词
    :param src_mail: Pandas.DataFrame
    :param lang: str
    :return: Pandas.DataFrame
    """
    mail_content = src_mail.Body[0]
    mail_ends = cfg.get('Email_Settings', 'Email_End_Str').split('|')
    for end in mail_ends:
        mail_content = mail_content.split(end, 1)[0]
    if lang in ['zh-cn', 'zh-tw']:
        terms = zh_to_terms(mail_content)
    elif lang in ['en']:
        terms = en_to_terms(mail_content)
    else:
        terms = ja_to_terms(mail_content)
    src_mail.Body[0] = terms
    if cfg.getboolean('Internal', 'Debug_Terms'):
        print("[Debug] {}".format(str(terms)))
    return src_mail

if __name__ == "__main__":
    src_mail = get_emails()
    # 判断获取原始邮件内容是否为空, 如果不为空, 则继续处理邮件内容
    if src_mail.Subject.count() != 0:
        src_mail, lang = chk_emails(src_mail)
        pre_mail = pre_emails_to_terms(src_mail, lang)
        # 分词列表
        terms = pre_mail.Body[0]
        # 对分词进行编码
        with open(os.path.join(base_dir, 'params', 'term_index.pkl'), 'rb') as file:
            term_index = pickle.load(file)
        term_data = torch.tensor(np.array(terms_to_encoding(term_index, pre_mail)), dtype=torch.int64)
        # 调换位置
        term_data = term_data.permute(1, 0)
        # 利用模型来获取预测结果
        model = RNN_LSTM_Net(len(term_index) + 1)
        model.load_state_dict(torch.load(os.path.join(base_dir, 'params', 'best_lstm_model')))
        # 更改模型为预测模式
        model.eval()
        y_pred_src = model(term_data)
        y_pred_max = torch.argmax(y_pred_src, dim=1).item()
        # 预留的调式信息
        if cfg.getboolean('Internal', 'Debug_Mail_Info'):
            print("[Debug] 邮件标题: {}".format(pre_mail.Subject[0]))
            print("[Debug] 发件姓名: {}".format(pre_mail.SenderName[0]))
            print("[Debug] 原始数值: {}".format(str(y_pred_src)))
            print("[Debug] 预测分类: {}".format(str(y_pred_max)))
            print("[Debug] 分词详情: {}".format(str(terms[:term_data.shape[0]])))
            print("[Debug] 分词编码: {}".format(str(np.array(term_data.permute(1, 0)))[1:-1]))
        # 如果预测的值为 0, 则弹出警告信息
        if y_pred_max == 0:
            if str(pre_mail.SenderName[0]).strip() not in eval(cfg.get('Filters', 'By_Email_SenderName_Not_in')):
                warn_content = "邮件标题: {}".format(pre_mail.Subject[0])
                win32api.MessageBox(0, warn_content, "Email 预警", win32con.MB_ICONWARNING)


