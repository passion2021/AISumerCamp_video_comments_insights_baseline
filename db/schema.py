from mongoengine import *
connect('xfyun', host='localhost', port=27017)

class Sentiment(Document):
    comment_id = StringField()
    comment_text = StringField()
    raw_result = DictField()
    sentiment_category = IntField()  # "1-正面","2-负面","3-正负混合","4-中性","5-不相关"
    user_scenario = IntField()  # "0-否","1-是"
    user_question = IntField()  # "0-否","1-是"
    user_suggestion = IntField()  # "0-否","1-是"

