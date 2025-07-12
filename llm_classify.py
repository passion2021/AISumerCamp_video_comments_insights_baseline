import pandas as pd
from libs.easy_llm import *

video_data = pd.read_csv("origin_videos_data.csv")
video_data["text"] = video_data["video_desc"].fillna("") + " " + video_data["video_tags"].fillna("")
prompt = """
你是一个专业的电商产品识别系统。请从视频描述中识别推广商品，必须严格选择以下选项之一：
- Xfaiyx Smart Translator
- Xfaiyx Smart Recorder

规则说明：
1. 当描述提到"翻译"、"多语言"等关键词时选择Translator
2. 当描述出现"录音"、"转写"等关键词时选择Recorder
3. 遇到不确定情况时选择更符合主要功能的选项

示例：
输入：这款设备支持实时语音转文字
输出：Xfaiyx Smart Recorder

现在请识别：
输入：{}

输出限制：
无需输出任何前缀后缀，以及任何解释，直接输出Xfaiyx Smart Translator或Xfaiyx Smart Recorder之一。
如果没有匹配的选项，也需要输出2个选项之中最相近的选项。
"""

def check_llm_output(llm_output):
    llm_output = llm_output.strip()
    if "Xfaiyx Smart Translator" in llm_output:
        return "Xfaiyx Smart Translator"
    elif "Xfaiyx Smart Recorder" in llm_output:
        return "Xfaiyx Smart Recorder"
    else:
        return None


for index, row in video_data.iterrows():
    print(f"text={row['text']}")
    if row['product_name'] == "Xfaiyx Smart Recorder" or row['product_name'] == "Xfaiyx Smart Translator":
        print(f"[跳过] index: {index}, Video ID: {row['video_id']}, Product Name: {row['product_name']}")
        continue
    messages = MsgList(HumanMsg(prompt.format(row['text'])))
    llm_resp = ''
    for chunk in qwen2_5_14b.stream(messages.to_json()):
        llm_resp += chunk

    classify_result = check_llm_output(llm_resp)

    video_data.at[index, 'product_name'] = classify_result

    print(f"index: {index}, Video ID: {row['video_id']}, Product Name: {row['product_name']}, Classify Result: {classify_result}")
    print()

submit_df = video_data[["video_id", "product_name"]]

submit_df.to_csv("submit_videos.csv", index=False)
print("✅ 保存完成：submit_videos.csv")
