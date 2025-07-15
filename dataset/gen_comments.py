import pandas as pd
from mongoengine import *
from db.schema import Sentiment
from loguru import logger
# 连接数据库
connect('xfyun', host='localhost', port=27017)

# 读取原始 CSV 文件
df_comments = pd.read_csv('../data/submit_comments.csv')

# 显示原始数据的列，查看是否包含 `comment_id`
print(df_comments.columns)


# 定义一个函数来获取数据库中的情感分析数据
def get_sentiment_data(comment_id):
    # 查询数据库中该 comment_id 的情感分析数据
    sentiment_data = Sentiment.objects(comment_id=comment_id).first()
    if sentiment_data:
        return sentiment_data.sentiment_category, sentiment_data.user_scenario, sentiment_data.user_question, sentiment_data.user_suggestion
    else:
        logger.error(f"没有找到 comment_id={comment_id} 的情感分析数据")
        raise Exception(f"没有找到 comment_id={comment_id} 的情感分析数据")


# 遍历 CSV 中的每一行，更新数据
for index, row in df_comments.iterrows():
    try:
        sentiment_category, user_scenario, user_question, user_suggestion = get_sentiment_data(row['comment_id'])
    except Exception as e:
        continue
    # 将数据库中的情感分析数据更新到 DataFrame 中
    df_comments.at[index, 'sentiment_category'] = sentiment_category
    df_comments.at[index, 'user_scenario'] = user_scenario
    df_comments.at[index, 'user_question'] = user_question
    df_comments.at[index, 'user_suggestion'] = user_suggestion

# 保存为新的 CSV 文件
df_comments.to_csv('../data/submit_comments_updated.csv', index=False,encoding='utf-8')

print("更新完成！已保存为 'submit_comments_updated.csv'")
