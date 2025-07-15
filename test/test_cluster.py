import pandas as pd
import os
import re

# 设置输入文件路径
input_path = r"E:\AISumerCamp_video_comments_insights_baseline\submit\submit_comments.csv"
output_dir = r"E:\AISumerCamp_video_comments_insights_baseline\submit\positive_clusters"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 加载 CSV
df = pd.read_csv(input_path)

# 只保留 sentiment_category 为 1 或 3 的评论
positive_df = df[df['sentiment_category'].isin([1, 3])]

# 删除空主题词行
# 第一步：去除缺失值（NaN）所在的行
positive_df = positive_df[positive_df['positive_cluster_theme'].notna()]
# 第二步：去除空字符串的主题词（即 ""）
positive_df = positive_df[positive_df['positive_cluster_theme'].str.strip() != ""]
# 第三步（可选）：去除仅包含分隔符的情况，比如 "|"
positive_df = positive_df[positive_df['positive_cluster_theme'].str.replace('|', '', regex=False).str.strip() != ""]

# 分组并输出
for theme, group in positive_df.groupby('positive_cluster_theme'):
    # 清理文件名（移除非法字符）
    safe_theme = re.sub(r'[\\/:*?"<>|]', '_', theme.replace('|', '_'))
    file_name = f"cluster_{safe_theme}.csv"
    output_path = os.path.join(output_dir, file_name)

    group.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"导出：{output_path}")
