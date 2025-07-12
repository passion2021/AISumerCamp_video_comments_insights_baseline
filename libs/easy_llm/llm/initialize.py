from .ollamachat import OllamaChat
from ..llm.deepseekchat import DeepSeekChat, AsyncDeepSeekChat
from ..llm.openaichat import OpenAIChat, AsyncOpenAIChat
from dotenv import load_dotenv
import os

load_dotenv()
deepseek_api_key = os.getenv('deepseek_api_key')
siliconflow_base_url = "https://api.siliconflow.cn/v1"
siliconflow_api_key = os.getenv('siliconflow_api_key')

deepseek_chat = DeepSeekChat(model="deepseek-chat", api_key=deepseek_api_key, base_url='https://api.deepseek.com/v1')
async_deepseek_chat = AsyncDeepSeekChat(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))
openai_chat = OpenAIChat(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
async_openai_chat = AsyncOpenAIChat(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), max_concurrent_tasks=10)
deepseek_v3_chat = DeepSeekChat(model="deepseek-ai/DeepSeek-V3",
                                api_key=siliconflow_api_key,
                                base_url=siliconflow_base_url)
deepseek_r1_chat = DeepSeekChat(model="deepseek-ai/DeepSeek-R1",
                                api_key=siliconflow_api_key,
                                base_url=siliconflow_base_url)
async_deepseek_v3_chat = AsyncDeepSeekChat(model="deepseek-ai/DeepSeek-V3",
                                           api_key=siliconflow_api_key,
                                           base_url=siliconflow_base_url)
async_deepseek_r1_chat = AsyncDeepSeekChat(model="deepseek-ai/DeepSeek-R1",
                                           api_key=siliconflow_api_key,
                                           base_url=siliconflow_base_url)
gemma3_4b = OllamaChat(model='gemma3:4b')
qwen2_5_14b = OllamaChat(model='qwen2.5:14b')
qwen3_8b = OllamaChat(model='qwen3:8b')
deepseek_r1_14b = OllamaChat(model='deepseek-r1:14b')
