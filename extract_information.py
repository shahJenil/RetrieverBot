from pymongo import MongoClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredImageLoader,
)
import gradio as gr
from gradio.themes.base import Base
import key_param
import os
import tempfile

client = MongoClient(
    key_param.MONGO_URI,
)
try:
    client.server_info()  # Test connection
    print("MongoDB connection established")
except Exception as e:
    raise Exception(f"Failed to connect to MongoDB: {e}")

dbname = "langchain"
clName = "mixed_data_samples"
collection = client[dbname][clName]

# Initialize embeddings model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=key_param.GOOGLE_API_KEY
)

vectorStore = MongoDBAtlasVectorSearch(
    collection=collection, embedding=embeddings, index_name="vector_index"
)


def get_loader_for_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == ".pdf":
        return PyPDFLoader(file_path)
    elif file_extension == ".txt":
        return TextLoader(file_path)
    elif file_extension == ".csv":
        return CSVLoader(file_path)
    elif file_extension in [".jpg", ".jpeg", ".png"]:
        return UnstructuredImageLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


def process_uploaded_file(uploaded_file):
    """Process a single uploaded file and add it to the vector store"""
    if isinstance(uploaded_file, str):
        temp_file_path = uploaded_file
    else:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
        ) as temp_file:
            with open(uploaded_file.name, "rb") as f:
                temp_file.write(f.read())
            temp_file_path = temp_file.name

    try:
        loader = get_loader_for_file(temp_file_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source"] = os.path.basename(temp_file_path)
            doc.metadata["file_type"] = os.path.splitext(temp_file_path)[1].lower()

        vectorStore = MongoDBAtlasVectorSearch.from_documents(
            documents,
            embeddings,
            collection=collection,
            index_name="vector_index",
        )
        return len(documents)
    except Exception as e:
        raise e
    finally:
        if not isinstance(uploaded_file, str) and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def query_data(query):
    """Query the database and get responses"""
    docs = vectorStore.similarity_search(query, k=1)
    as_output = docs[0].page_content if docs else "No relevant documents found"
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest", api_key=key_param.GOOGLE_API_KEY
    )
    retriever = vectorStore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
    retriever_output = qa.run(query)
    return as_output, retriever_output


def upload_file(file_obj):
    try:
        if file_obj is None:
            return "No file uploaded"
        file_extension = os.path.splitext(file_obj.name)[1].lower()
        allowed_extensions = [".txt", ".pdf", ".csv", ".jpg", ".jpeg", ".png"]
        if file_extension not in allowed_extensions:
            return f"Unsupported file type: {file_extension}. Please upload .txt, .pdf, .csv, or image files (.jpg, .png)."
        num_docs = process_uploaded_file(file_obj)
        return f"Successfully processed {file_obj.name}. Added {num_docs} document(s) to the database."
    except Exception as e:
        return f"Error processing file: {str(e)}"


# Gradio UI
with gr.Blocks(
    theme=Base(), title="Question Answering App Using Vector Search + RAG"
) as demo:
    gr.Markdown(
        "# Question Answering App Using Atlas Vector Search And Retrieval Augmented Architecture"
    )
    with gr.Tab("Upload Files"):
        gr.Markdown("### Upload files to add to the knowledge base")
        file_input = gr.File(
            label="Upload a file (.txt, .pdf, .csv)",
            file_types=[".txt", ".pdf", ".csv"],
        )
        upload_button = gr.Button("Upload", variant="primary")
        upload_output = gr.Textbox(label="Upload Status", interactive=False)
        upload_button.click(upload_file, inputs=[file_input], outputs=[upload_output])
    with gr.Tab("Ask Questions"):
        textbox = gr.Textbox(label="Enter your Question")
        button = gr.Button("Submit", variant="primary")
        with gr.Column():
            output1 = gr.Textbox(
                lines=1,
                max_lines=10,
                label="Output with just Atlas Vector Search (returns the provided text field):",
            )
            output2 = gr.Textbox(
                lines=1,
                max_lines=10,
                label="Output generated by chaining Atlas Vector Search to LangChain's RetrievalQA and Gemini LLM:",
            )
        button.click(query_data, textbox, outputs=[output1, output2])

if __name__ == "__main__":
    demo.launch()
