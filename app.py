import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.openai import OpenAI

load_dotenv()

def main():

    st.set_page_config(page_title='Chatting with PDFs')
    st.header("Let's Chat")
    pdf=st.file_uploader('Upload your PDF here',type=['pdf'])
    if pdf is not None:
        reader=PdfReader(pdf)
        text=""
        for page in reader.pages:
            text=text + page.extract_text() + '\n'
        

        splitting_text=CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=250,
            length_function=len
        )
        chunks=splitting_text.split_text(text)

        


        embeddings=OpenAIEmbeddings()
        knowledge_base=FAISS.from_texts(chunks,embeddings)

        user_question= st.text_input('Ask a question:')
        docs=knowledge_base.similarity_search(user_question)

        

        llm=OpenAI()
        chain=load_qa_chain(llm,chain_type='stuff')
        response=chain.run(input_documents=docs,question=user_question)
        st.write(response)
    


if __name__ == '__main__':
    main()
