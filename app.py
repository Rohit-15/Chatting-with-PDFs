import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate

openai_api_key = st.secrets["openai"]["api_key"]

def main():
    st.set_page_config(page_title='Chatting with PDFs')
    st.header("Let's Chat")

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
        
        st.success("PDF uploaded and processed successfully!")
        
        user_question = st.text_input('Ask a question about the PDF:')
        
        if user_question:
            if not user_question.strip():
                st.warning("Please enter a valid question.")
            else:
                try:
                    docs = knowledge_base.similarity_search(user_question,k=5, threshold=0.75)

                    custom_prompt = PromptTemplate(
                        input_variables=["context", "question"],
                        template="Given the context below,answer the question. If answer is not in the context, say Please Ask a Question pertaining to the PDF.:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:")

                    
                    llm = OpenAI(openai_api_key=openai_api_key)
                    chain = load_qa_chain(llm, chain_type='stuff',prompt=custom_prompt)
                    response = chain.run(input_documents=docs, question=user_question)
                    
                    st.write("Answer:")
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload a PDF to start asking questions.")

if __name__ == '__main__':
    main()
