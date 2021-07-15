__version__ = '0.1.0'

'''
Hi TalkBot~
Author:RedElectricity
'''

#导入相关模块
from fuzzywuzzy import fuzz #模糊查询
import jieba #结巴分词
import pickle #持久化
import httpx #网络交互
import yaml #配置文件需要的模块
import random

#导入设置
config = yaml.load(open("config,yaml",'r'),Loader=yaml.FullLoader())

async def Talk(UserID:int ,Message:str):
    #取出用户数据
    user_data = pickle.load(open('message.bin','rb'))
    #检查是否是新用户
    if UserID in user_data:
        if len(user_data[UserID]) > 1:
            #判断是否是已经回答过的问题
            if fuzz(Message,user_data[UserID][len(user_data[UserID])-1]) >= 80:
               return random.sample(config['Wording']['dwta'],1)[0] 
            else:
                #这里是回答参数
                pass
    else:
        user_data[UserID] = []
        user_data[UserID].append(Message)
