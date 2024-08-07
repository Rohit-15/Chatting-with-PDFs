import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.openai import OpenAI

openai_api_key = st.secrets["openai"]["api_key"]

def main():
    st.set_page_config(page_title='Chatting with PDFs')
    st.header("Let's Chat with Your PDF")
    pdf = st.file_uploader('Upload your PDF here', type=['pdf'])

    if pdf is not None:
        reader = PdfReader(pdf)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + '\n'
        
        splitting_text = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=250,
            length_function=len
        )
        chunks = splitting_text.split_text(text)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        user_question = st.text_input('Ask a question:')
        if user_question:
            docs = knowledge_base.similarity_search(user_question, k=5)  # Retrieve top 5 similar chunks

            # Check if there are any relevant documents based on a similarity threshold (optional)
            # If using a similarity threshold, ensure docs include similarity scores and filter accordingly
            if docs:
                llm = OpenAI(openai_api_key=openai_api_key)
                chain = load_qa_chain(llm, chain_type='stuff')

                # Explicitly state in the prompt to answer based on the provided documents
                response = chain.run(input_documents=docs, question=user_question)
                st.write(response)
            else:
                st.write("The question may not be relevant to the provided document.")

if __name__ == '__main__':
    main()
