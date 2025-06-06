import os
import json
from openai import OpenAI
from glob import glob
from pymilvus import model as milvus_model
from pymilvus import MilvusClient
from tqdm import tqdm

# # 设置HuggingFace镜像站
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 
# os.environ['HUGGINGFACE_CO_URL_HOME'] = 'https://hf-mirror.com'

# 加载deepseek模型
api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1",  # DeepSeek API 的基地址
)
# print(api_key)

# 加载embedding 模型
embedding_model = milvus_model.DefaultEmbeddingFunction()
print(embedding_model)

# 创建数据库
milvus_client = MilvusClient(uri="milvus_demo.db")
collection_name = "my_rag_collection"
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)

milvus_client.create_collection(
    collection_name=collection_name,
    dimension=768,
    metric_type="IP",
    consistency_level="Strong")


# 加载文本数据
text_lines = []

for file_path in glob(r"mfd.md", recursive=True):
    with open(file_path, "r") as file:
        file_text = file.read()

    text_lines += file_text.split("# ")
print(len(text_lines))


# 文本数据保存到数据库
data = []
doc_embeddings = embedding_model.encode_documents(text_lines)

for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
    data.append({"id": i, "vector": doc_embeddings[i], "text": line})

milvus_client.insert(collection_name=collection_name, data=data)


# 问题检索
question1 = "房子买了以后土地归谁所有？"

search_res = milvus_client.search(
    collection_name=collection_name,
    data=embedding_model.encode_queries(
        [question1]
    ),  # 将问题转换为嵌入向量
    limit=3,  # 返回前3个结果
    search_params={"metric_type": "IP", "params": {}},  # 内积距离
    output_fields=["text"],  # 返回 text 字段
)

retrieved_lines_with_distances = [
    (res["entity"]["text"], res["distance"]) for res in search_res[0]
]
# print(json.dumps(retrieved_lines_with_distances, indent=4))


# 将检索结果写进提示词

context = "\n".join(
    [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
)

SYSTEM_PROMPT = """
Human: 你是一个 AI 助手。你能够从提供的上下文段落片段中找到问题的答案。
"""
USER_PROMPT = f"""
请使用以下用 <context> 标签括起来的信息片段来回答用 <question> 标签括起来的问题。
<context>
{context}
</context>
<question>
{question1}
</question>
"""

# 回答问题

response = deepseek_client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
)
print("deepseek的回答：")
print(response.choices[0].message.content)



#  第二个问题
question2 = "土地承包的期限？"

search_res = milvus_client.search(
    collection_name=collection_name,
    data=embedding_model.encode_queries(
        [question2]
    ),  # 将问题转换为嵌入向量
    limit=3,  # 返回前3个结果
    search_params={"metric_type": "IP", "params": {}},  # 内积距离
    output_fields=["text"],  # 返回 text 字段
)

retrieved_lines_with_distances = [
    (res["entity"]["text"], res["distance"]) for res in search_res[0]
]

context = "\n".join(
    [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
)

USER_PROMPT = f"""
请使用以下用 <context> 标签括起来的信息片段来回答用 <question> 标签括起来的问题。
<context>
{context}
</context>
<question>
{question2}
</question>
"""

response = deepseek_client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
)
print("deepseek的回答：")
print(response.choices[0].message.content)


"""
deepseek的回答：
根据提供的上下文，虽然其中包含了关于地役权、权利质权以及合同变更和转让的相关法律规定，但并未直接涉及房屋购买后土地所有权归属的问题。  

在中国，根据《中华人民共和国土地管理法》和《中华人民共和国物权法》的相关规定（虽然这些法律未在提供的上下文中直接引用），土地所有权属于国家或集体所有，个人或单位只能取得土地使用权（如建设用地使用权、宅基地使用权等）。因此，购买房屋后：  

1. **土地所有权**：仍归国家（城市土地）或集体（农村土地）所有，购房者无法取得土地所有权。  
2. **土地使用权**：购房者通过房屋转让取得相应的土地使用权（如住宅用地的建设用地使用权），其期限和权利范围需依据土地出让合同或法律规定。  

若需更具体的法律依据，建议参考《中华人民共和国土地管理法》《中华人民共和国城市房地产管理法》等法律法规中关于土地所有权与使用权的规定。


deepseek的回答：
在提供的上下文中，并未提及关于土地承包期限的具体规定。当前上下文主要涉及权利质权、合同变更转让以及地役权的相关条款，但未包含与土地承包期限相关的法律条文。建议查阅其他法律条文（如《农村土地承包法》）以获取该问题的具体答案。

"""