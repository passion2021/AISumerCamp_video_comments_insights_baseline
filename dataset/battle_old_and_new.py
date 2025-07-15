import pandas as pd


new_df = pd.read_csv('../data/submit_comments.csv')
print(new_df.columns)
for i,row in new_df.iterrows():
    print(i, row['positive_cluster_theme'], row['negative_cluster_theme'], row['scenario_cluster_theme'], row['question_cluster_theme'], row['suggestion_cluster_theme'])
    print()