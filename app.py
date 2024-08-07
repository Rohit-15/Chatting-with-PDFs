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
                    docs = knowledge_base.similarity_search(user_question, k=5, score_threshold=0.7)
                    
                    if docs:
                        llm = OpenAI(
                            openai_api_key=openai_api_key,
                            model_name="gpt-3.5-turbo",
                            temperature=0.3,
                            max_tokens=150
                        )
                        prompt_template = """You are an AI assistant that answers questions based solely on the given context. If the answer cannot be found in the context, say "I don't have enough information to answer that question based on the PDF content." Do not use any external knowledge.

                        Context: {context}

                        Question: {question}

                        Answer: """
                        PROMPT = PromptTemplate(
                            template=prompt_template, input_variables=["context", "question"]
                        )
                        chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
                        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                        answer = response['output_text']
                        if answer.strip():
                            st.write("Answer:")
                            st.write(answer)
                        else:
                            st.warning("The answer to your question is not found in the PDF.")
                    else:
                        st.warning("No relevant information found in the PDF.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload a PDF to start asking questions.")
