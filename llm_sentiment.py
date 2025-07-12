import json

import pandas as pd
from libs.easy_llm import *
from libs.easy_llm.utils.extract_tools import json_parse_dirty

df_comments = pd.read_csv('./origin_comments_data.csv')
prompt = '''
请对以下评论进行多维度分析。

分析规则：
1. 用户场景判断：评论提及使用场合/环境（如"出差用"、"上课时"）则标1
2. 用户疑问判断：包含问号或疑问词（吗/怎么/为什么）标1
3. 情感混合判断：同时出现褒贬词（如"质量好但价格高"）选3

评论样本：{}
'''
output = '''
输出限制：
无需输出任何前缀后缀，以及任何解释。严格按照格式返回JSON。
{
  "sentiment": ["1-正面","2-负面","3-正负混合","4-中性","5-不相关"],
  "has_scenario": [0/1],
  "has_question": [0/1],
  "has_suggestion": [0/1]
}
'''
# print(df_comments.columns)
for index, row in df_comments.iterrows():
    messages = MsgList(HumanMsg(prompt.format(row['comment_text']) + output))
    llm_resp = ''
    for chunk in qwen2_5_14b.stream(messages.to_json()):
        llm_resp += chunk
    result = json_parse_dirty(llm_resp)
    print(f"index: {index}, Comment ID: {row['comment_id']}, Comment Text: {row['comment_text']}, Result: {result}")
