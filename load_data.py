from pymongo import MongoClient
# from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.llms import openai
from langchain.chains import retrieval_qa
import gradio as gr
from gradio.themes.base import Base
import key_param

client = MongoClient(key_param.MONGO_URI)
dbname = "langchain"
clName = "text_samples"
collection = client[dbname][clName]

loader = DirectoryLoader('./sample_files', glob="./*.txt", show_progress=True)
data = loader.load()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=key_param.openai_api_key)

vectorStore = MongoDBAtlasVectorSearch.from_documents(data,embeddings,collection=collection)

