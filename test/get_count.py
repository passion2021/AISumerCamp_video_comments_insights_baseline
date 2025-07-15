import pandas as pd
from db.schema import Sentiment
old_df = pd.read_csv('../data/copy/submit_comments.csv')


for i,row in old_df.iterrows():
    print(i, row['comment_id'])

# 一共6476个数据
