import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Streamlit app layout
st.title("📚 PDF Q&A with LangChain & Streamlit")

system_template = r'''
Use the following pieces of context to answer the user's question.
Before answering translate your response to English.
If you don't find the answer in the provided context, just respond "I don't know."
---------------
Context: ```{context}```
'''

user_template = '''
Question: ```{question}```
'''

messages= [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(user_template)
]

qa_prompt = ChatPromptTemplate.from_messages(messages)







uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Save the uploaded file temporarily
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load the PDF document
    st.write(f"Loading the document: {uploaded_file.name}")
    loader = PyPDFLoader(uploaded_file.name)
    data = loader.load()
    
    # Chunk the document
    st.write("Splitting the document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    
    # Generate embeddings and vector store
    st.write("Generating embeddings and creating a vector store...")
    embeddings = HuggingFaceEmbeddings()  
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Setup conversational retriever chain
    groq_api_key = os.getenv('GROQ_API_KEY')
    lm = ChatGroq(groq_api_key=groq_api_key, model='llama-3.1-70b-versatile', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    crc = ConversationalRetrievalChain.from_llm(
        llm=lm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': qa_prompt },
        chain_type='stuff',
        verbose=False
    )
    
    # User question input
    question = st.text_input("Ask a question about the PDF:")
    
    if question:
        # Get the answer from the ConversationalRetrievalChain
        st.write("Retrieving the answer...")
        answer = crc({"question": question})  # Change 'query' to 'question'
        
        # Display the answer
        st.write("### Answer:")
        st.write(answer['answer'])


# Add a footer or some instructions if needed
st.write("Upload a PDF and start asking questions!")
