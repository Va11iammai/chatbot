from pathlib import Path
import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv() 

groq_api_key = os.environ['GROQ_API_KEY']

llm_groq = ChatGroq(
            groq_api_key=groq_api_key, model_name="llama3-70b-8192",
                         temperature=0.2)


@cl.on_chat_start
async def on_chat_start():

    msg1 = cl.Message("Welcome to Research-Enhanced Augmented Conversational Technology! I'm here to provide research-driven, insightful assistance, powered by advanced language models tailored to your needs")
    await msg1.send()

    file_paths = [str(file) for file in Path("docs").rglob('*') if file.is_file()]
                  
    texts = []
    metadatas = []
    for file_path in file_paths:
        pdf = PyPDF2.PdfReader(file_path)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        file_texts = text_splitter.split_text(pdf_text)
        texts.extend(file_texts)

        file_metadatas = [{"source": f"{i}-{file_path.split('/')[-1]}"} for i in range(len(file_texts))]        
        metadatas.extend(file_metadatas)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )
    
    message_history = ChatMessageHistory()
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_groq,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    
    msg2 = cl.Message(content=f"Files processed. You can now ask questions!")
    await msg2.send()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler()
    
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"] 

    text_elements = [] 
    
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]
        
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"
    await cl.Message(content=answer, elements=text_elements).send()