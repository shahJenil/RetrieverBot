from pymongo import MongoClient
# from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.llms import openai
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
import key_param

client = MongoClient(key_param.MONGO_URI)
dbname = "langchain"
clName = "text_samples"
collection = client[dbname][clName]

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=key_param.openai_api_key)

vectorStore = MongoDBAtlasVectorSearch(collection, embeddings)

def query_data(query):
    docs = vectorStore.similarity_search(query,k=1)
    as_output = docs[0].page_content

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=key_param.openai_api_key)
    retriever = vectorStore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm,chain_type="stuff",retriever=retriever)
    retriever_output = qa.run(query)

    return as_output,retriever_output

with gr.Blocks(theme=Base(), title = "Question Answering App Using Vector Search + RAG") as demo:
    gr.Markdown(
        """
        # Question Answering App Using Atlas Vector Search And Retrieval Augmented Architecture
        """
    )
    textbox = gr.Textbox(label="Enter your Question")
    with gr.Row():
        button = gr.Button("Submit", variant="primary")
    with gr.Column():
        output1 = gr.Textbox(lines=1,max_lines=10, label="Output with just Atlas Vector Search (returns the provided text field) :")
        output2 = gr.Textbox(lines=1,max_lines=10, label="Output generated by chaining Atlas Vector Search to LangChain's RetrievalQA and Gemini LLM :")
    
    button.click(query_data,textbox, outputs=[output1,output2])

demo.launch()