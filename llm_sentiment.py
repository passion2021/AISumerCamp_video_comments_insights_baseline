import json
from loguru import logger
import pandas as pd

from db.schema import Sentiment
from libs.easy_llm import *
from libs.easy_llm.utils.extract_tools import json_parse_dirty

# 创建 logs 文件夹（如果不存在）
os.makedirs("logs", exist_ok=True)

# 设置 loguru 输出到文件，保存在 logs 目录
logger.add("logs/app.log",
           rotation="10 MB",  # 每个日志文件最大 10MB，自动切分
           retention="10 days",  # 日志保留 10 天
           compression="zip",  # 历史日志压缩为 zip
           encoding="utf-8")  # 避免中文乱码

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


def get_int_from_result(result, key, default=0):
    value = result.get(key, [default])
    if isinstance(value, list):
        return int(value[0])
    return int(value)


def save_sentiment_to_mongo(result: dict, row: pd.Series):
    try:
        sentiment_map = {
            "1-正面": 1,
            "2-负面": 2,
            "3-正负混合": 3,
            "4-中性": 4,
            "5-不相关": 5
        }
        sentiment_label = result.get("sentiment")
        if isinstance(sentiment_label, list):
            sentiment_label = sentiment_label[0]
        sentiment_value = sentiment_map.get(sentiment_label, None)

        doc = Sentiment(
            comment_id=row["comment_id"],
            comment_text=row["comment_text"],
            raw_result=result,
            sentiment_category=sentiment_value,
            user_scenario=get_int_from_result(result, "has_scenario"),
            user_question=get_int_from_result(result, "has_question"),
            user_suggestion=get_int_from_result(result, "has_suggestion")
        )
        doc.save()
        logger.info(f"保存成功：comment_id={row['comment_id']}")
    except Exception as e:
        logger.error(f"保存失败：comment_id={row['comment_id']}，错误：{e}")


def call_llm(row):
    messages = MsgList(HumanMsg(prompt.format(row['comment_text']) + output))
    llm_resp = ''
    for chunk in qwen2_5_14b.stream(messages.to_json()):
        llm_resp += chunk
    return json_parse_dirty(llm_resp)

# 11.17 -> 14:05
print(df_comments.columns)
for index, row in df_comments.iterrows():
    # 检查 MongoDB 是否已有记录
    if Sentiment.objects(comment_id=row['comment_id']).first():
        logger.info(f"[跳过] index: {index}, Comment ID: {row['comment_id']} 已存在，跳过分析")
        continue

    result = call_llm(row)
    if not result:
        logger.warning(f"index: {index}, Comment ID: {row['comment_id']}, json_parse_dirty error!")
    logger.info(
        f"index: {index}, Comment ID: {row['comment_id']}, Comment Text: {row['comment_text']}, Result: {result}")

    save_sentiment_to_mongo(result, row)
