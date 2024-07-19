from langchain.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os


text_field = "text"

# switch back to normal index for langchain
index_name = "modelpy"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model_name = "text-embedding-ada-002"

embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)

vectorstore: PineconeVectorStore = PineconeVectorStore.from_existing_index(
    index_name=index_name, embedding=embed
)

query = "What is Modelpy"

# completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0.0
)

qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
)

print(qa.invoke(query))
