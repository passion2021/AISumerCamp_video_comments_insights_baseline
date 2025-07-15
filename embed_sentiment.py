from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
import pandas as pd

from libs.qwen_embed import QwenEmbeddingTransformer

comments_data = pd.read_csv('./origin_comments_data.csv')
for col in ['sentiment_category',
            'user_scenario', 'user_question', 'user_suggestion']:
    qwen_transformer = QwenEmbeddingTransformer()
    predictor = make_pipeline(
        qwen_transformer,
        SGDClassifier(max_iter=1000, random_state=42)  # 增加迭代次数确保收敛
    )

    # 训练数据（过滤掉product_name为空的样本）
    train_texts = comments_data[~comments_data[col].isnull()]["comment_text"]
    train_labels = comments_data[~comments_data[col].isnull()][col]

    # 训练模型
    predictor.fit(train_texts, train_labels)

    # 对所有视频数据的text进行预测（注意处理空文本或无效输入）
    comments_data[col] = predictor.predict(comments_data["comment_text"])



# 保存到新的 CSV 文件中
comments_data.to_csv('./predicted_comments_data.csv', index=False, encoding='utf-8-sig')
print("✅ 已保存为：predicted_comments_data.csv")