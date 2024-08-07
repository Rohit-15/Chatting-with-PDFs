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
    st.header("Let's Chat")

    pdf = st.file_uploader('Upload your PDF here', type=['pdf'])
    
    if pdf is not None:
        reader = PdfReader(pdf)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + '\n'
        
        splitting_text = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,  # Adjusted chunk size for better segmentation
            chunk_overlap=200,  # Adjusted overlap for smoother transitions
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
                    docs = knowledge_base.similarity_search(user_question, k=5)
                    
                    if not docs:
                        st.warning("No relevant information found in the PDF.")
                    else:
                        # Filter documents based on relevance to the question
                        relevant_docs = [doc for doc in docs if is_relevant(doc, user_question)]
                        
                        if len(relevant_docs) > 0:
                            llm = OpenAI(openai_api_key=openai_api_key)
                            chain = load_qa_chain(llm, chain_type='specific_to_pdf_content')  # Use a more specific QA chain
                            response = chain.run(input_documents=relevant_docs, question=user_question)
                            
                            if response.strip():
                                st.write("Answer:")
                                st.write(response)
                            else:
                                st.warning("The answer to your question is not found in the PDF.")
                        else:
                            st.warning("No relevant documents found after filtering.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload a PDF to start asking questions.")

def is_relevant(document, question):
    # Implement logic to determine if a document is relevant to the question
    # This could involve comparing the document's content with the question
    pass

if __name__ == '__main__':
    main()
