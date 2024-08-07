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

            if docs:  # If documents are found
                # You can add a check for minimum similarity score here if needed
                llm = OpenAI(openai_api_key=openai_api_key)
                chain = load_qa_chain(llm, chain_type='stuff')

                response = chain.run(input_documents=docs, question=user_question)
                if response.strip():
                    st.write(response)
                else:
                    st.write("This question is not relevant to the content present in the PDF.")
            else:
                st.write("This question is not relevant to the content present in the PDF.")

if __name__ == '__main__':
    main()
