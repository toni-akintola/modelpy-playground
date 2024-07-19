import os
from pinecone import Pinecone, ServerlessSpec
from datasets import load_dataset
import pandas as pd
from tabulate import tabulate
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import time
from tqdm.auto import tqdm
from uuid import uuid4
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore

load_dotenv()

create_doc_path = "/Users/Toni/The Vault/modelpy/modelpy-web/content/docs/create.mdx"
index_doc_path = "/Users/Toni/The Vault/modelpy/modelpy-web/content/docs/index.mdx"
installation_doc_path = (
    "/Users/Toni/The Vault/modelpy/modelpy-web/content/docs/installation.mdx"
)
package_path = "/Users/Toni/The Vault/modelpy/modelpy-web/content/docs/package.mdx"

dataset = load_dataset(
    "text",
    data_files={
        "docs": [create_doc_path, index_doc_path, installation_doc_path, package_path]
    },
    split="docs[:10000]",
)


# initialize connection to pinecone (get API key at app.pinecone.io)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# configure client
pc = Pinecone(api_key=PINECONE_API_KEY)


# create the length function
tokenizer = tiktoken.get_encoding("cl100k_base")


def tiktoken_len(text):
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


tiktoken_len(
    "hello I am a chunk of text and using the tiktoken_len function "
    "we can find the length of this chunk of text in tokens"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""],
)

model_name = "text-embedding-ada-002"

embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)

spec = ServerlessSpec(cloud="aws", region="us-east-1")

index_name = "modelpy"
existing_indices = [index_info["name"] for index_info in pc.list_indexes()]

# check if index already exists (it shouldn't if this is first time)
if index_name not in existing_indices:
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=1536,  # dimensionality of ada 002
        metric="dotproduct",
        spec=spec,
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)
time.sleep(1)
# view index stats
index.describe_index_stats()
texts = []
metadatas = []
batch_limit = 100
for record in dataset.to_iterable_dataset():
    record_texts = text_splitter.split_text(record["text"])
    metadata = {"created_at": time.time()}
    texts.extend(record_texts)

    record_metadatas = [
        {"chunk": j, "text": text, **metadata} for j, text in enumerate(record_texts)
    ]
    metadatas.extend(record_metadatas)

    # if we have reached the batch_limit we can add texts
    if len(texts) >= batch_limit:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))
        texts = []
        metadatas = []


if len(texts) > 0:
    ids = [str(uuid4()) for _ in range(len(texts))]
    embeds = embed.embed_documents(texts)
    index.upsert(vectors=zip(ids, embeds, metadatas))
