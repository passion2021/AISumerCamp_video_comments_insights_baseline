import math

import pandas as pd

df_video = pd.read_csv('../origin_videos_data.csv')
df_comments = pd.read_csv('../origin_comments_data.csv')
# for index, row in df_video.iterrows():
#     # print(row['video_id'], row['video_desc'], row['video_tags'], row['product_name'],sep=' | ')
#     print(index, row['product_name'])

one_print = 0
count = 0
for index, row in df_comments.iterrows():
    count += 1
    if count > 2000:
        break

    sentiment = row.tolist()[3:7]
    keyword = row.tolist()[7:]
    if not one_print and all(math.isnan(x) for x in sentiment):
        one_print = 1
        print(f'最后一条标注数据索引位置：{count-1}')

    if not one_print:
        print(count, "情感分析：", sentiment, '关键词聚类：', keyword)

#
# row_1741 = df_comments.iloc[1740]
# print(row_1741)