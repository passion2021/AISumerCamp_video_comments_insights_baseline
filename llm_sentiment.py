import pandas as pd

df_comments = pd.read_csv('./origin_comments_data.csv')
prompt = '''
请对以下评论进行多维度分析。

分析规则：
1. 用户场景判断：评论提及使用场合/环境（如"出差用"、"上课时"）则标1
2. 用户疑问判断：包含问号或疑问词（吗/怎么/为什么）标1
3. 情感混合判断：同时出现褒贬词（如"质量好但价格高"）选3

评论样本：{}

输出限制：
严格按照格式返回JSON。
{
  "sentiment": ["1-正面","2-负面","3-正负混合","4-中性","5-不相关"],
  "has_scenario": [0/1],
  "has_question": [0/1],
  "has_suggestion": [0/1]
}
'''
print(df_comments.columns)
for index, row in df_comments.iterrows():
    print(index, row.tolist())
