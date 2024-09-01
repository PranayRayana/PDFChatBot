#import all libraries

from dotenv import load_dotenv  # Load environment variables from a .env file
import streamlit as st # Streamlit for creating web apps
import os  # Provides functions to interact with the operating system
import google.generativeai as genai # Google Generative AI module
import asyncio # Asyncio for handling asynchronous operations
from PyPDF2 import PdfReader  # PDF reader for extracting text from PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter  # Text splitters for handling large text (RecussiveCharacterTextSplitter is used in this project)
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI # Google Generative AI embeddings and chat models
from langchain_community.vectorstores.faiss import FAISS # FAISS for vector storage and similarity search
from langchain.chains.question_answering import load_qa_chain  # Load question-answering chain
from langchain.prompts import PromptTemplate  # Prompt template for formatting prompts


load_dotenv() # Load environment variables

#configuring api_key. The key is placed in .env file. 
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


#using PyPDF, reading the pdf documents and extracting the text from it
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text


#The text extracted from the PyPDF are splitted into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter( chunk_size = 10000, chunk_overlap = 1000) 
    chunks=text_splitter.split_text(text)
    return chunks


#converting the chunks into vector embeddings and storing it. (creating vector store or knowledge base)
def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks, embedding = embeddings ) #FAISS is used here. we can use chromaDB or vectorDB as well
    vector_store.save_local("faiss_index")
    return vector_store



# Define the prompt template for question answering
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in the provided context,
    Just say "answer is not aavailable in the context", don't provide wrong information and blank response\n\n
    Context=\n{context}?\n
    Question=\n{question}\n 
    Answer:
    """
     # Load the Google Generative AI chat model
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt=PromptTemplate(template=prompt_template, input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff", prompt = prompt)

    return chain


def user_input(user_question):
    # Perform similarity search on the vector store based on the user's question
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_load = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs=vector_load.similarity_search(user_question)
    # Get the conversational chain and generate a response
    chain = get_conversational_chain()
    response = chain(
        {"input_documents":docs, "question" : user_question}
        , return_only_outputs=True )
    print(response)
    st.write("Reply: ", response["output_text"])


def get_or_create_eventloop():
    # Get or create a new event loop for handling async operations
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

# Create a new event loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

def main():
    # Streamlit page configuration and main UI logic
    st.set_page_config("PDF Chatbot") # Set page title
    st.header("PDF Chatbot") # display the header

    user_question = st.text_input("Enter your question", label_visibility="visible")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        # Upload PDF files in the sidebar menu
        st.title("Document upload Section")
        pdf_docs = st.file_uploader("Upload PDF files here", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done") 

# Run the main function when the script is executed
if __name__ == "__main__":
    main()