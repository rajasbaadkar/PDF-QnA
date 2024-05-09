import os
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import pickle
from dotenv import load_dotenv


#Sidebar
with st.sidebar:
    st.title('PDF QA App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built for PDF query answering.
    ''')
    add_vertical_space(5)
    st.write('Made by [Rajas Baadkar](https://github.com/rajasbaadkar)')

#Body
def main():
    st.title('Chat with AI for PDF Queries 💬')
    st.write('Tired of reading the monotonous and boring PDFs?')
    st.write("Don't worry, we got you!")
    st.write("")
    st.write("")
    st.write("")
    load_dotenv()       #This will load the OpenAI API Key from .env file

    #Upload PDF
    st.subheader("Upload your PDF below:")
    file = st.file_uploader("",type='pdf')

    #Read only if uploaded
    if file is not None:
        #Read PDF
        pdf_reader = PdfReader(file)
        st.write(pdf_reader)
    
        st.write('\n')
        #Extracting text from PDF
        st.subheader('PDF Content')
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        #Splitting PDF into chunks
        text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                        )
        
        chunks = text_splitter.split_text(text=text)

        #Creating Vector Embeddings
        store_name = file.name[:-4]     #To get name of file without the .pdf extension

        #Check if the pickle file is already present in disk, if yes then read it
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                vectorstore = pickle.load(f)
            st.write('Embeddings Loaded from Disk')
        #If not present, write into the disk a new pickle file and create embeddings
        
        else:
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(chunks,embedding = embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(vectorstore,f)    #write the vectorstore in disk

            #st.write('New Embeddings Loaded into Disk')
        
        #User Question/Query
        question = st.text_input("Ask your query regarding the PDF:")
        st.write(question)

        if question: 
            #Performs a semantic search in the vectorstore comparing with the question
            docs = vectorstore.similarity_search(query = question, k=3)     #k is context window
            llm = OpenAI(model_name='gpt-3.5-turbo',temperature=0) 
            chain = load_qa_chain(llm=llm,chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs,question=question)
                print(cb)    
            st.write(response)

if __name__ == '__main__':
    main()