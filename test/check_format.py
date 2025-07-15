import chardet
import pandas as pd

comments = 'E:\AISumerCamp_video_comments_insights_baseline\data\submit_comments.csv'
videos = 'E:\AISumerCamp_video_comments_insights_baseline\data\submit_videos.csv'

df_1 = pd.read_csv(comments, encoding='utf-8')  # 或根据需要调整编码
df_2 = pd.read_csv(videos, encoding='utf-8')


df_1.to_csv(comments, index=False, encoding='utf-8')
df_2.to_csv(videos, index=False, encoding='utf-8')