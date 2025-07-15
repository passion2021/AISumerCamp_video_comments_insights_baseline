import os
import zipfile
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def sentiment_classify(comments_data: pd.DataFrame):
    for col in ['sentiment_category',
                'user_scenario', 'user_question', 'user_suggestion']:
        predictor = make_pipeline(
            TfidfVectorizer(tokenizer=jieba.lcut), SGDClassifier()
        )
        predictor.fit(
            comments_data[~comments_data[col].isnull()]["comment_text"],
            comments_data[~comments_data[col].isnull()][col],
        )
        comments_data[col] = predictor.predict(comments_data["comment_text"])
    return comments_data


def find_best_k(embeddings, min_k=5, max_k=8):
    best_k = min_k
    best_score = -1
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        if len(set(labels)) > 1:  # 避免单一簇报错
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_k, best_score = k, score
    return best_k


def cluster_comments(data, text_column='comment_text', filter_condition=None, output_column=None, stop_words=None):
    try:
        target_data = data[filter_condition].copy()
        if len(target_data) < 5:  # 样本过少跳过
            return data

        # 优化分词（添加商品名词典）
        jieba.add_word("Xfaiyx Smart Translator")
        jieba.add_word("Xfaiyx Smart Recorder")

        tfidf = TfidfVectorizer(
            tokenizer=lambda x: [w for w in jieba.lcut(x) if w not in stop_words],
            max_features=1000,
            stop_words=stop_words
        )
        embeddings = tfidf.fit_transform(target_data[text_column])

        best_k = find_best_k(embeddings)
        kmeans = KMeans(n_clusters=best_k, random_state=42).fit(embeddings)

        # 提取主题词（过滤停用词后取top5）
        feature_names = tfidf.get_feature_names_out()
        cluster_top_words = []
        for i in range(best_k):
            top_indices = kmeans.cluster_centers_[i].argsort()[-10:][::-1]
            keywords = [feature_names[idx] for idx in top_indices
                        if feature_names[idx] not in stop_words]
            cluster_top_words.append(", ".join(keywords[:5]))

        data.loc[target_data.index, output_column] = [
            cluster_top_words[label] for label in kmeans.labels_
        ]
    except Exception as e:
        print(f"聚类失败（{output_column}）: {str(e)}")
    return data


def execute_cluster(comments_data: pd.DataFrame):
    STOP_WORDS = ["的", "了", "啊", "呢", "这个", "那个"]
    CLUSTER_CONFIGS = [
        {'filter': comments_data['sentiment_category'].isin([1, 3]), 'output_col': 'positive_cluster_theme'},
        {'filter': comments_data['sentiment_category'].isin([2, 3]), 'output_col': 'negative_cluster_theme'},
        {'filter': comments_data['user_scenario'] == 1, 'output_col': 'scenario_cluster_theme'},
        {'filter': comments_data['user_question'] == 1, 'output_col': 'question_cluster_theme'},
        {'filter': comments_data['user_suggestion'] == 1, 'output_col': 'suggestion_cluster_theme'}
    ]

    for config in CLUSTER_CONFIGS:
        print(f"正在生成 {config['output_col']}...")
        comments_data = cluster_comments(
            data=comments_data,
            text_column='comment_text',
            filter_condition=config['filter'],
            output_column=config['output_col'],
            stop_words=STOP_WORDS
        )
    return comments_data


def generate_zip(video_data: pd.DataFrame, comments_data: pd.DataFrame):
    # 检查并创建目录
    if not os.path.exists("submit"):
        os.makedirs("submit")
    # 保存视频数据 CSV 文件
    video_data[["video_id", "product_name"]].to_csv("submit/submit_videos.csv", index=None, encoding='utf-8')
    # 保存评论数据 CSV 文件
    comments_data[['video_id', 'comment_id', 'sentiment_category',
                   'user_scenario', 'user_question', 'user_suggestion',
                   'positive_cluster_theme', 'negative_cluster_theme',
                   'scenario_cluster_theme', 'question_cluster_theme',
                   'suggestion_cluster_theme']].to_csv("submit/submit_comments.csv", index=None, encoding='utf-8')
    # 创建一个 Zip 文件并将 CSV 文件添加到其中（保留 submit/ 目录结构）
    with zipfile.ZipFile("submit/submit.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write("submit/submit_videos.csv", arcname="submit/submit_videos.csv")
        zipf.write("submit/submit_comments.csv", arcname="submit/submit_comments.csv")

    print("✅ submit.zip 已创建，结构如下：")
    print("submit/")
    print("  ├─ submit_videos.csv")
    print("  └─ submit_comments.csv")


if __name__ == '__main__':
    comments_data = pd.read_csv("origin_comments_data.csv")
    comments_data = sentiment_classify(comments_data)
    comments_data = execute_cluster(comments_data)
    video_data = pd.read_csv("data/copy/submit_videos.csv")

    generate_zip(video_data, comments_data)
